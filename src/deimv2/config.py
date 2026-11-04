"""DEIMv2-S settings. Use a separate environment for the official implementation."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATASET = ROOT / "data/finetune_hrp4k"
REPO_DIR = ROOT / "out/vendor/DEIMv2"
PYTHON = REPO_DIR / ".venv/bin/python"
OUTPUT_DIR = ROOT / "out/deimv2-small-finetuned"
EPOCHS = 100
RESOLUTION = 896
BATCH_SIZE = 32
WORKERS = 2
LEARNING_RATE = 5e-4
FREEZE_BACKBONE = True
SEED = 42
