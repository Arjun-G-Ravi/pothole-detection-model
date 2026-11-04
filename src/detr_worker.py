"""Run an official detector in its own Python process/environment; no cross-model imports."""

import argparse
import copy
import csv
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEIM_REPO = "Intellindust/DEIMv2_DINOv3_S_COCO"


def checkpoint_state(checkpoint):
    if checkpoint.get("ema") is not None:
        return checkpoint["ema"]["module"]
    return checkpoint.get("model", checkpoint)


def matching_state(model_state, pretrained):
    """Keep compatible features and regression weights; initialize the new pothole heads normally."""
    matched = {key: value for key, value in pretrained.items()
               if key in model_state and value.shape == model_state[key].shape}
    missing_features = [key for key in model_state if key.startswith("backbone.") and key not in matched]
    if missing_features:
        raise ValueError(f"Pretrained backbone does not match this architecture: {missing_features[:5]}")
    return matched


def public_pretrained(kind, cfg):
    import torch

    cache = ROOT / "out/pretrained"
    cache.mkdir(parents=True, exist_ok=True)
    if kind == "dfine":
        path = cache / "dfine_s_coco.pth"
        if not path.is_file():
            url = "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_s_coco.pth"
            temporary = path.with_suffix(".download")
            torch.hub.download_url_to_file(url, str(temporary))
            temporary.replace(path)
    else:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        # Public model only; do not use stored Hugging Face credentials.
        options = {"repo_id": DEIM_REPO, "token": False}
        architecture = json.loads(Path(hf_hub_download(filename="config.json", **options)).read_text())
        # The authors' HF checkpoint extracts different ViT layers than their original YAML.
        cfg["DINOv3STAs"].update(architecture["DINOv3STAs"])
        cfg["DINOv3STAs"]["weights_path"] = None
        path = cache / "deimv2_s_coco_hf.pth"
        if not path.is_file():
            weights = hf_hub_download(filename="model.safetensors", **options)
            torch.save({"model": load_file(weights)}, path)
    return path


def keep_backbone_in_eval(model, inputs):
    model.backbone.eval()


def replace_scheduler_config(cfg, requested):
    # Upstream recursively merges dictionaries; changing scheduler type must replace its arguments too.
    cfg.yaml_cfg["lr_scheduler"] = copy.deepcopy(requested["lr_scheduler"])


def export_metrics(run_dir):
    """Make the official JSONL training history convenient to compare with the other models."""
    log = run_dir / "log.txt"
    if not log.is_file():
        return
    rows = []
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        metrics = row.get("test_coco_eval_bbox", [])
        rows.append({"epoch": row["epoch"], "loss": row.get("train_loss"),
                     "mAP50_95": metrics[0] if metrics else None,
                     "mAP50": metrics[1] if len(metrics) > 1 else None,
                     "mAP75": metrics[2] if len(metrics) > 2 else None})
    if rows:
        with (run_dir / "metrics.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)


def train(args, metadata, core, solver_module, dist):
    import hashlib
    import torch
    import yaml

    run_dir = args.run_dir
    # Detect changed source annotations before a resumed run can silently use different data.
    for details in metadata["dataset"].values():
        digest = hashlib.sha256(Path(details["source_annotations"]).read_bytes()).hexdigest()
        if digest != details["sha256"]:
            raise ValueError("Prepared source annotations changed since this run was created.")
    config_path = run_dir / ("resolved_config.yaml" if args.resume else "config.yaml")
    config = yaml.safe_load(config_path.read_text())
    pretrained = None
    if not args.resume:
        pretrained = metadata["pretrained"]
        if pretrained is None:
            pretrained = public_pretrained(metadata["kind"], config)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    dist.setup_distributed(seed=metadata["seed"])
    cfg = core.YAMLConfig(str(config_path), device=args.device, resume=str(args.resume) if args.resume else None,
                          tuning=None, use_amp=args.device.startswith("cuda"))
    replace_scheduler_config(cfg, config)
    # Save a fully expanded config so later predictions do not depend on mutable upstream defaults.
    resolved = dict(cfg.yaml_cfg)
    resolved.pop("__include__", None)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    revision = subprocess.run(["git", "-C", metadata["repo_dir"], "rev-parse", "HEAD"],
                              capture_output=True, text=True)
    metadata["source_revision"] = revision.stdout.strip() if revision.returncode == 0 else "unversioned"
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2))
    model = cfg.model
    if pretrained:
        state = checkpoint_state(torch.load(pretrained, map_location="cpu", weights_only=False))
        matched = matching_state(model.state_dict(), state)
        model.load_state_dict(matched, strict=False)
        print(f"Loaded {len(matched)} compatible pretrained tensors; new class heads initialized separately.")
    if metadata["freeze_backbone"]:
        model.backbone.requires_grad_(False)
        model.register_forward_pre_hook(keep_backbone_in_eval)
        print("Frozen backbone; training encoder/projector, decoder, and detection heads.")
    solver = solver_module.TASKS[cfg.yaml_cfg["task"]](cfg)
    try:
        solver.fit()
    finally:
        export_metrics(run_dir)
        dist.cleanup()


def prediction_rows(labels, boxes, scores, confidence):
    return sorted(
        [{"label": "pothole", "score": float(score), "box_xyxy": [float(value) for value in box]}
         for label, box, score in zip(labels, boxes, scores) if int(label) == 0 and float(score) >= confidence],
        key=lambda row: row["score"], reverse=True,
    )


def predict(args, metadata, core):
    import torch
    from PIL import Image, ImageOps
    from torchvision.transforms import functional as functional

    config_path = args.run_dir / "resolved_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing resolved training config: {config_path}")
    cfg = core.YAMLConfig(str(config_path))
    if cfg.yaml_cfg.get("num_classes") != 1 or cfg.yaml_cfg.get("remap_mscoco_category"):
        raise ValueError("Prediction requires a one-class pothole checkpoint from these training scripts.")
    model = cfg.model
    state = checkpoint_state(torch.load(args.checkpoint, map_location="cpu", weights_only=False))
    model.load_state_dict(state, strict=True)
    model = model.to(args.device).eval()
    postprocessor = cfg.postprocessor.to(args.device).eval()
    request = json.loads(args.request.read_text())
    predictions = []
    for index, path in enumerate(request["paths"]):
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
        resolution = metadata["resolution"]
        tensor = functional.to_tensor(functional.resize(image, [resolution, resolution]))
        if metadata["kind"] == "deimv2":
            tensor = functional.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # The official postprocessors expect width, height, and return original-image xyxy boxes.
        sizes = torch.tensor([image.size], device=args.device)
        with torch.inference_mode():
            result = postprocessor(model(tensor.unsqueeze(0).to(args.device)), sizes)[0]
        detections = prediction_rows(result["labels"].cpu().tolist(), result["boxes"].cpu().tolist(),
                                    result["scores"].cpu().tolist(), request["confidence"])
        predictions.append(detections)
        print(f"{metadata['kind']}: {index + 1}/{len(request['paths'])}", flush=True)
    args.output.write_text(json.dumps(predictions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "predict"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metadata = json.loads((args.run_dir / "run.json").read_text())
    repo = Path(metadata["repo_dir"])
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import torch

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    prefix = "src" if metadata["kind"] == "dfine" else "engine"
    core = importlib.import_module(f"{prefix}.core")
    if args.action == "train":
        train(args, metadata, core, importlib.import_module(f"{prefix}.solver"),
              importlib.import_module(f"{prefix}.misc.dist_utils"))
    else:
        if not all((args.checkpoint, args.request, args.output)):
            parser.error("Prediction needs --checkpoint, --request, and --output.")
        predict(args, metadata, core)


if __name__ == "__main__":
    main()
