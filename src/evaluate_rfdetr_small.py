"""Compare the base pothole model and fine-tuned checkpoints on four splits."""

import argparse
import csv
import gc
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
from tqdm import tqdm

from pothole_data import DATA_DIR, HRP4K_DIR, HRP4K_REPO, ROOT
from plot_evaluation import create_plots


# Edit this folder after training.
MODEL_DIR = ROOT / "out/hrp4k-rfdetr-small-finetuned/2026-10-25_19-56-17_610318"

CHECKPOINTS = [
    "checkpoint_10.ckpt",
    "checkpoint_20.ckpt",
    "checkpoint_24.ckpt",
    "checkpoint_best_total.pth",
]
BASE_WEIGHTS = None  # Optional local base checkpoint; None uses Hugging Face.
BASE_REPO = "dronefreak/hrp4k-rfdetr-small"
DATASET_DIR = DATA_DIR / "finetune_hrp4k"
RESOLUTION = 896
CONFIDENCE = 0.20  # Precision, recall, and F1 use this fixed threshold.
IOU_THRESHOLD = 0.50
MAX_DETECTIONS = 100  # Standard COCO maximum per image.
TRAIN_MY_DATASET_FRACTION = 1.0
TRAIN_HRP_FRACTION = .1
VAL_MY_DATASET_FRACTION = 1.0  # 1.0 = all, 0.25 = 25%.
VAL_HRP_FRACTION = .15
VALIDATION_SEED = 42


def evaluation_subset(samples, fraction):
    """Select the same evaluation images for every checkpoint."""
    if not 0 < fraction <= 1:
        raise ValueError("Evaluation fractions must be greater than 0 and at most 1.")
    if fraction == 1 or not samples:
        return samples
    count = max(1, round(len(samples) * fraction))
    indices = sorted(random.Random(VALIDATION_SEED).sample(range(len(samples)), count))
    return [samples[index] for index in indices]


def load_prepared_split(split, local_names):
    """Use the actual COCO splits prepared for training, including empty images."""
    folder = DATASET_DIR / split
    coco = json.loads((folder / "_annotations.coco.json").read_text())
    pothole_ids = {category["id"] for category in coco["categories"]
                   if category["name"].lower() == "pothole"}
    boxes = defaultdict(list)
    for annotation in coco["annotations"]:
        if annotation["category_id"] in pothole_ids:
            boxes[annotation["image_id"]].append(annotation["bbox"])

    local, hrp = [], []
    for image in coco["images"]:
        path = folder / image["file_name"]
        if not path.is_file():
            raise FileNotFoundError(path)
        # Training saves images as <image_id>_<original_filename>.
        original_name = Path(image["file_name"]).name.split("_", 1)[1]
        destination = local if original_name in local_names else hrp
        destination.append((path, boxes[image["id"]]))
    return local, hrp


def load_hrp_validation():
    """Use HRP4K's official validation split, downloading only missing files."""
    from huggingface_hub import hf_hub_download, snapshot_download

    metadata_file = "data/images/valid/metadata.jsonl"
    metadata_path = HRP4K_DIR / metadata_file
    if not metadata_path.is_file():
        hf_hub_download(repo_id=HRP4K_REPO, repo_type="dataset",
                        filename=metadata_file, local_dir=str(HRP4K_DIR))
    with metadata_path.open() as file:
        rows = [json.loads(line) for line in file if line.strip()]
    rows = evaluation_subset(rows, VAL_HRP_FRACTION)
    filenames = [f"data/images/valid/{row['file_name']}" for row in rows]
    missing = [filename for filename in filenames if not (HRP4K_DIR / filename).is_file()]
    if missing:
        print(f"Downloading {len(missing)} HRP4K validation images...")
        snapshot_download(repo_id=HRP4K_REPO, repo_type="dataset",
                          allow_patterns=missing, local_dir=str(HRP4K_DIR))
    return [(HRP4K_DIR / filename, row["objects"]["bbox"])
            for filename, row in zip(filenames, rows)]


def load_splits():
    coco = json.loads((DATA_DIR / "annotations/original_coco.json").read_text())
    local_names = {Path(image["file_name"]).name for image in coco["images"]}
    train_local, train_hrp = load_prepared_split("train", local_names)
    val_local, _ = load_prepared_split("valid", local_names)
    splits = {
        "train_my_dataset": evaluation_subset(train_local, TRAIN_MY_DATASET_FRACTION),
        "val_my_dataset": evaluation_subset(val_local, VAL_MY_DATASET_FRACTION),
        "train_hrp": evaluation_subset(train_hrp, TRAIN_HRP_FRACTION),
        "val_hrp": load_hrp_validation(),
    }
    for name, samples in splits.items():
        if not samples:
            raise ValueError(f"No images in {name}. Check the prepared training dataset.")
        for path, _ in samples:
            if not path.is_file():
                raise FileNotFoundError(path)
    return splits


def match_predictions(predicted_boxes, target_boxes):
    """Mark true positives with confidence-ordered, one-to-one IoU matching."""
    correct = torch.zeros(len(predicted_boxes), dtype=torch.bool)
    if len(predicted_boxes) and len(target_boxes):
        overlaps = box_iou(predicted_boxes, target_boxes)
        matched = torch.zeros(len(target_boxes), dtype=torch.bool)
        for index, overlap in enumerate(overlaps):
            overlap[matched] = -1
            best_iou, target_index = overlap.max(dim=0)
            if best_iou >= IOU_THRESHOLD:
                matched[target_index] = True
                correct[index] = True
    return correct


def confidence_curves(scores, correct, total_targets):
    """Sweep thresholds without repeating inference or box matching."""
    thresholds = torch.arange(101, dtype=torch.float32) / 100
    order = scores.argsort(descending=True)
    counts = torch.searchsorted(-scores[order], -thresholds, right=True)
    cumulative_tp = torch.cat([torch.zeros(1, dtype=torch.int64), correct[order].cumsum(0)])
    tp = cumulative_tp[counts]
    precision = tp / counts.clamp(min=1)
    recall = tp / max(total_targets, 1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    return {
        "confidence": thresholds.tolist(),
        "precision": (precision * 100).tolist(),
        "recall": (recall * 100).tolist(),
        "F1": (f1 * 100).tolist(),
        "detections": counts.tolist(),
    }


def evaluate_split(model, model_name, split_name, samples):
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox",
                                  backend="faster_coco_eval")
    all_scores, all_correct = [], []
    total_targets = 0
    for path, boxes in tqdm(samples, desc=f"{model_name} / {split_name}", unit="image"):
        with Image.open(path) as original:
            image = original.convert("RGB")
        with torch.inference_mode():
            # AP needs the entire confidence ranking, not a display threshold.
            detections = model.predict(image, threshold=0.0)
        potholes = torch.tensor(
            [str(name).lower() == "pothole" for name in detections.data["class_name"]],
            dtype=torch.bool,
        )
        predicted_boxes = torch.as_tensor(detections.xyxy, dtype=torch.float32)[potholes]
        scores = torch.as_tensor(detections.confidence, dtype=torch.float32)[potholes]
        order = scores.argsort(descending=True)[:MAX_DETECTIONS]
        predicted_boxes, scores = predicted_boxes[order], scores[order]

        target_boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        target_boxes[:, 2:] += target_boxes[:, :2]  # COCO xywh -> xyxy.
        prediction = {"boxes": predicted_boxes, "scores": scores,
                      "labels": torch.zeros(len(scores), dtype=torch.int64)}
        target = {"boxes": target_boxes,
                  "labels": torch.zeros(len(target_boxes), dtype=torch.int64)}
        metric.update([prediction], [target])
        all_scores.append(scores)
        all_correct.append(match_predictions(predicted_boxes, target_boxes))
        total_targets += len(target_boxes)

    ap = metric.compute()
    scores, correct = torch.cat(all_scores), torch.cat(all_correct)
    selected = scores >= CONFIDENCE
    tp = int(correct[selected].sum())
    fp = int(selected.sum()) - tp
    fn = total_targets - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    row = {
        "model": model_name,
        "split": split_name,
        "images": len(samples),
        "potholes": tp + fn,
        "mAP50": float(ap["map_50"]) * 100 if ap["map_50"] >= 0 else None,
        "mAP50_95": float(ap["map"]) * 100 if ap["map"] >= 0 else None,
        "mAP75": float(ap["map_75"]) * 100 if ap["map_75"] >= 0 else None,
        "mAR100": float(ap["mar_100"]) * 100 if ap["mar_100"] >= 0 else None,
        "precision": precision * 100,
        "recall": recall * 100,
        "F1": 200 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "TP": tp, "FP": fp, "FN": fn,
    }
    return row, confidence_curves(scores, correct, total_targets)


def save_results(run_dir, rows, curves, settings):
    with (run_dir / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "results.json").write_text(json.dumps({"settings": settings, "results": rows}, indent=2))
    (run_dir / "curves.json").write_text(json.dumps(curves))
    with (run_dir / "curves.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "split", "confidence", "precision", "recall", "F1", "detections"])
        for model_name, splits in curves.items():
            for split_name, curve in splits.items():
                for values in zip(*(curve[key] for key in
                                    ("confidence", "precision", "recall", "F1", "detections"))):
                    writer.writerow([model_name, split_name, *values])


def print_results(rows):
    print("\nAll metric values are percentages. P/R/F1 use the fixed confidence and IoU thresholds.")
    width = max(20, max(len(row["model"]) for row in rows))
    print(f"{'Model':<{width}} {'Split':<18} {'Images':>6} {'mAP50':>8} {'mAP50-95':>9} {'Prec.':>8} {'Recall':>8} {'F1':>8}")
    for row in rows:
        values = [f"{row[key]:.2f}" if row[key] is not None else "N/A"
                  for key in ("mAP50", "mAP50_95", "precision", "recall", "F1")]
        print(f"{row['model']:<{width}} {row['split']:<18} {row['images']:>6} "
              f"{values[0]:>8} {values[1]:>9} {values[2]:>8} {values[3]:>8} {values[4]:>8}")


def discover_checkpoints(folder, names=None):
    """Use the requested list, or discover all checkpoints in numerical order."""
    if names is not None:
        paths = [folder / name for name in dict.fromkeys(names)]
        if not paths:
            raise ValueError("CHECKPOINTS is empty. Set it to None to evaluate all checkpoints.")
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Selected checkpoint not found: {path}")
            if path.suffix not in {".pth", ".ckpt"}:
                raise ValueError(f"Expected a .pth or .ckpt checkpoint: {path}")
        return paths

    def order(path):
        epoch = path.stem.removeprefix("checkpoint_")
        return (0, int(epoch)) if epoch.isdigit() else (1, path.name)

    return sorted((path for path in folder.iterdir()
                   if path.is_file() and path.suffix in {".pth", ".ckpt"}), key=order)


def main(plots_only=False):
    model_dir = MODEL_DIR.expanduser().resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"Model directory not found: {model_dir}\nEdit MODEL_DIR at the top of this script.")
    if plots_only:
        saved = sorted(path.parent for path in (model_dir / "evaluation").glob("*/results.json"))
        if not saved:
            raise SystemExit("No saved evaluation results in this model folder. Run evaluation first.")
        create_plots(saved[-1], model_dir)
        return

    from huggingface_hub import hf_hub_download
    from rfdetr import RFDETRSmall

    paths = discover_checkpoints(model_dir, CHECKPOINTS)
    if not paths:
        raise SystemExit(f"No .pth or .ckpt files found in {model_dir}")
    base = Path(BASE_WEIGHTS).expanduser().resolve() if BASE_WEIGHTS else Path(hf_hub_download(
        repo_id=BASE_REPO, filename="checkpoint_best_total.pth",
    ))
    checkpoints = {"base": base, **{path.name: path for path in paths}}
    splits = load_splits()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = model_dir / "evaluation" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    settings = {
        "resolution": RESOLUTION, "device": device,
        "confidence_for_precision_recall_f1": CONFIDENCE,
        "iou_for_precision_recall_f1": IOU_THRESHOLD,
        "ap_confidence_floor": 0.0, "max_detections_per_image": MAX_DETECTIONS,
        "metric_units": "percent", "checkpoints": {name: str(path) for name, path in checkpoints.items()},
        "confidence_sweep": "0.00 to 1.00 in steps of 0.01, at the configured IoU threshold",
        "training_fractions": {"train_my_dataset": TRAIN_MY_DATASET_FRACTION, "train_hrp": TRAIN_HRP_FRACTION},
        "validation_fractions": {"val_my_dataset": VAL_MY_DATASET_FRACTION, "val_hrp": VAL_HRP_FRACTION},
        "validation_seed": VALIDATION_SEED,
        "split_image_counts": {name: len(samples) for name, samples in splits.items()},
        "hrp_validation": "Sampled from the official HRP4K valid split, never from training images.",
    }
    (run_dir / "splits.json").write_text(json.dumps({
        name: [{"path": str(path), "boxes_xywh": boxes} for path, boxes in samples]
        for name, samples in splits.items()
    }))
    print(f"Evaluating the base model and {len(paths)} checkpoints. Results: {run_dir}")
    print(f"Device: {device}; resolution: {RESOLUTION}; confidence: {CONFIDENCE}; IoU: {IOU_THRESHOLD}")
    rows, curves = [], {}
    for name, checkpoint in checkpoints.items():
        # Explicit settings also recover the stripped base checkpoint's configuration.
        model = RFDETRSmall.from_checkpoint(checkpoint, device=device, resolution=RESOLUTION,
                                           num_select=300, dec_layers=3)
        if "pothole" not in [label.lower() for label in model.class_names]:
            raise ValueError(f"{name}: checkpoint has no pothole class: {model.class_names}")
        curves[name] = {}
        for split_name, samples in splits.items():
            row, curve = evaluate_split(model, name, split_name, samples)
            rows.append(row)
            curves[name][split_name] = curve
            save_results(run_dir, rows, curves, settings)
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    print_results(rows)
    create_plots(run_dir, model_dir)
    print(f"\nSaved evaluation results and graphs to {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots-only", action="store_true",
                        help="Regenerate graphs from the latest saved evaluation without model inference.")
    main(parser.parse_args().plots_only)
