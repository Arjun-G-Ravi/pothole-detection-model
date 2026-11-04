"""Edit these settings, then run finetune_yolo26_small.py without arguments."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATASET = ROOT / "data/finetune_hrp4k"
DATASET_DIR = ROOT / "data/finetune_yolo26"
OUTPUT_DIR = ROOT / "out/yolo26-small-finetuned"
WEIGHTS = "yolo26s.pt"  # Official pretrained weights, fetched when training starts.

TRAIN_SETTINGS = {
    "epochs": 100,
    "imgsz": 896,
    "batch": 64,
    "nbs": 32,  # One optimizer update per batch.
    "optimizer": "AdamW",
    "lr0": 1e-3,
    "lrf": 0.01,
    "cos_lr": False,
    "warmup_epochs": 3.0,
    "warmup_bias_lr": 0.0,
    "patience": 10,
    "freeze": 11,  # YOLO26 backbone modules 0 through 10.
    "seed": 42,
    "workers": 2,
    "multi_scale": 0.0,
    "save": True,
    "save_period": 1,
    "plots": True,
}
