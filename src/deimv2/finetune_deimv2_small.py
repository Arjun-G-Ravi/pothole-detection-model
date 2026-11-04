"""Fine-tune DEIMv2 Small; edit config.py, then run this script."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deimv2 import config
from detr_training import launch_training


if __name__ == "__main__":
    launch_training("deimv2", config)
