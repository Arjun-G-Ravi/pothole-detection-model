"""Shared local dataset/config preparation for the official D-FINE and DEIMv2 engines."""

import argparse
import copy
import hashlib
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIGS = {
    "dfine": "configs/dfine/dfine_hgnetv2_s_coco.yml",
    "deimv2": "configs/deimv2/deimv2_dinov3_s_coco.yml",
}


def prepare_annotations(source, destination):
    """Keep every COCO entry and box, changing only the pothole category ID to zero."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Generated annotations must be outside the source dataset.")
    prepared, manifest = {}, {}
    for split in ("train", "valid"):
        annotation_path = source / split / "_annotations.coco.json"
        raw = annotation_path.read_bytes()
        coco = json.loads(raw)
        ids = {row["id"] for row in coco["categories"] if row["name"].strip().lower() == "pothole"}
        if not ids:
            raise ValueError(f"No pothole category in {annotation_path}")
        images = {row["id"]: row for row in coco["images"]}
        if not images or len(images) != len(coco["images"]):
            raise ValueError(f"Empty dataset or duplicate image IDs in {annotation_path}")
        for row in images.values():
            image_path = (source / split / row["file_name"]).resolve()
            if not image_path.is_relative_to(source / split) or not image_path.is_file():
                raise FileNotFoundError(f"Prepared dataset image unavailable: {image_path}")
        converted = copy.deepcopy(coco)
        converted["categories"] = [{"id": 0, "name": "pothole", "supercategory": "road-damage"}]
        converted["annotations"] = []
        for annotation in coco["annotations"]:
            if annotation["category_id"] not in ids:
                continue
            image = images.get(annotation["image_id"])
            if image is None:
                raise ValueError(f"Annotation references an unknown image: {annotation}")
            box = annotation["bbox"]
            if len(box) != 4 or not all(math.isfinite(value) for value in box):
                raise ValueError(f"Invalid box: {box}")
            x, y, width, height = box
            if (min(x, y) < -1e-6 or min(width, height) <= 0
                    or x + width > image["width"] + 1e-6 or y + height > image["height"] + 1e-6):
                raise ValueError(f"Box outside image bounds: {box}")
            item = dict(annotation, category_id=0)
            item.setdefault("iscrowd", 0)
            item.setdefault("area", width * height)
            converted["annotations"].append(item)
        prepared[split] = converted
        manifest[split] = {"source_annotations": str(annotation_path),
                           "sha256": hashlib.sha256(raw).hexdigest(),
                           "images": len(coco["images"]), "boxes": len(converted["annotations"])}
    # Finish validation before writing anything. Source images and annotations are never edited.
    destination.mkdir(parents=True, exist_ok=True)
    for split, coco in prepared.items():
        (destination / f"{split}.json").write_text(json.dumps(coco), encoding="utf-8")
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def transforms(resolution, training, normalize):
    ops = [{"type": "RandomPhotometricDistort", "p": 0.5},
           {"type": "RandomHorizontalFlip"}] if training else []
    ops += [{"type": "Resize", "size": [resolution, resolution]},
            {"type": "ConvertPILImage", "dtype": "float32", "scale": True}]
    if normalize:
        ops.append({"type": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})
    if training:
        ops.append({"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True})
    return {"type": "Compose", "ops": ops,
            "policy": {"name": "default", "epoch": 100000000, "ops": []}}


def build_config(kind, settings, run_dir):
    epochs, resolution = settings.EPOCHS, settings.RESOLUTION
    cfg = {"__include__": [str(settings.REPO_DIR / MODEL_CONFIGS[kind])],
           "num_classes": 1, "remap_mscoco_category": False,
           "output_dir": str(run_dir), "eval_spatial_size": [resolution, resolution],
           "checkpoint_freq": 1, "print_freq": 20, "sync_bn": False,
           "use_ema": True, "use_amp": False, "use_wandb": False,
           "epochs" if kind == "dfine" else "epoches": epochs,
           "optimizer": {"type": "AdamW", "lr": settings.LEARNING_RATE,
                         "weight_decay": 1e-4, "params": []},
           "lr_scheduler": {"type": "CosineAnnealingLR", "T_max": epochs,
                            "eta_min": settings.LEARNING_RATE * 0.01},
           "lr_warmup_scheduler": {"type": "LinearWarmup", "warmup_duration": 100}}
    for split, key in (("train", "train_dataloader"), ("valid", "val_dataloader")):
        cfg[key] = {"total_batch_size": settings.BATCH_SIZE, "num_workers": settings.WORKERS,
                    "shuffle": split == "train", "drop_last": False,
                    "dataset": {"type": "CocoDetection", "img_folder": str(settings.SOURCE_DATASET / split),
                                "ann_file": str(run_dir / "dataset" / f"{split}.json"),
                                "return_masks": False,
                                "transforms": transforms(resolution, split == "train", kind == "deimv2")}}
    # Disable multi-scale and the upstream second-stage restart so every epoch saves a resumable checkpoint.
    cfg["train_dataloader"]["collate_fn"] = {
        "type": "BatchImageCollateFunction", "base_size": resolution,
        "base_size_repeat": None, "stop_epoch": epochs + 1,
    }
    if kind == "dfine":
        cfg["HGNetv2"] = {"pretrained": False}
    else:
        cfg.update({"DINOv3STAs": {"weights_path": None}, "lrsheduler": None,
                    "DEIMCriterion": {"matcher": {"change_matcher": False}}})
        cfg["train_dataloader"]["collate_fn"].update({"mixup_prob": 0.0, "copyblend_prob": 0.0})
    return cfg


def launch_training(kind, settings):
    parser = argparse.ArgumentParser(description=f"Fine-tune {kind} Small on the existing RF-DETR/YOLO splits.")
    parser.add_argument("--prepare-only", action="store_true", help="Validate data and write a run config; no imports or downloads.")
    parser.add_argument("--resume", type=Path, help="Resume last.pth or an epoch checkpoint in its original run folder.")
    parser.add_argument("--weights", type=Path, help="Local pretrained checkpoint instead of the public default.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if min(settings.EPOCHS, settings.BATCH_SIZE) <= 0 or settings.WORKERS < 0:
        parser.error("Epochs/batch must be positive and workers must be nonnegative.")
    if not math.isfinite(settings.LEARNING_RATE) or settings.LEARNING_RATE <= 0:
        parser.error("Learning rate must be finite and positive.")
    if settings.RESOLUTION <= 0 or settings.RESOLUTION % 32:
        parser.error("Resolution must be a positive multiple of 32.")
    if args.resume and args.weights:
        parser.error("Choose either --resume or --weights.")
    for path in (args.resume, args.weights):
        if path and not path.expanduser().is_file():
            parser.error(f"Checkpoint not found: {path}")
    if args.resume:
        checkpoint = args.resume.expanduser().resolve()
        run_dir = checkpoint.parent
        metadata = json.loads((run_dir / "run.json").read_text())
        if metadata["kind"] != kind:
            parser.error(f"This run belongs to {metadata['kind']}, not {kind}.")
    else:
        run_dir = settings.OUTPUT_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        manifest = prepare_annotations(settings.SOURCE_DATASET, run_dir / "dataset")
        cfg = build_config(kind, settings, run_dir)
        # JSON is valid YAML and avoids requiring ML/YAML dependencies for preparation.
        (run_dir / "config.yaml").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        metadata = {"kind": kind, "repo_dir": str(settings.REPO_DIR), "python": str(settings.PYTHON),
                    "resolution": settings.RESOLUTION, "freeze_backbone": settings.FREEZE_BACKBONE,
                    "seed": settings.SEED, "dataset": manifest,
                    "pretrained": str(args.weights.expanduser().resolve()) if args.weights else None}
        (run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Run directory: {run_dir}", flush=True)
    for split, details in metadata["dataset"].items():
        print(f"{split}: {details['images']} entries, {details['boxes']} boxes", flush=True)
    if args.prepare_only:
        return
    python, repo = Path(metadata["python"]), Path(metadata["repo_dir"])
    model_config = repo / MODEL_CONFIGS[kind]
    if not model_config.is_file():
        raise SystemExit(f"Official {kind} source code is missing: {model_config}\n"
                         "Installing src/detr_requirements.txt only installs dependencies. "
                         f"Extract the official source ZIP into {repo}, preserving .venv/; "
                         "configs/ must be directly inside that directory. "
                         "See docs/additional_models.md. No model was downloaded or trained.")
    if not python.is_file():
        raise SystemExit(f"Python executable is missing: {python}\n"
                         "Set PYTHON in the model's config.py to its installed environment, "
                         "or follow docs/additional_models.md to create it and install dependencies. "
                         "No model was downloaded or trained.")
    command = [str(python), str(ROOT / "src/detr_worker.py"), "train", "--run-dir", str(run_dir),
               "--device", args.device]
    if args.resume:
        command += ["--resume", str(checkpoint)]
    subprocess.run(command, cwd=repo, check=True)


def prediction_command(kind, checkpoint, request_path, output_path, device):
    checkpoint = Path(checkpoint).resolve()
    metadata = json.loads((checkpoint.parent / "run.json").read_text())
    if metadata["kind"] != kind:
        raise ValueError(f"Expected {kind}, but this checkpoint belongs to {metadata['kind']}")
    return [metadata["python"], str(ROOT / "src/detr_worker.py"), "predict",
            "--run-dir", str(checkpoint.parent), "--checkpoint", str(checkpoint),
            "--request", str(request_path), "--output", str(output_path), "--device", device]
