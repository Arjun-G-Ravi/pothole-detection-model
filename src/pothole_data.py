"""Dataset paths and loaders shared by training and the viewer."""

import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
HRP4K_DIR = DATA_DIR / "hrp4k_original"
HRP4K_REPO = "dronefreak/HRP4K"
HRP4K_IMAGES = 2000
SEED = 42


def load_local_samples():
    """Read boxes already annotated on our preprocessed images."""
    coco = json.loads((DATA_DIR / "annotations" / "original_coco.json").read_text())
    pothole_ids = {
        category["id"]
        for category in coco["categories"]
        if category["name"].lower() == "pothole"
    }
    if not pothole_ids:
        raise ValueError("The local COCO annotations have no pothole category.")

    boxes = defaultdict(list)
    for annotation in coco["annotations"]:
        if annotation["category_id"] in pothole_ids:
            boxes[annotation["image_id"]].append(annotation["bbox"])

    samples = []
    for image in coco["images"]:
        path = DATA_DIR / image["file_name"]
        if not path.is_file():
            raise FileNotFoundError(path)
        samples.append((path, boxes[image["id"]]))
    return samples


def load_hrp4k_samples():
    """Reuse the seeded HRP4K subset locally, downloading only missing files."""
    from huggingface_hub import hf_hub_download, snapshot_download

    metadata_file = "data/images/train/metadata.jsonl"
    metadata_path = HRP4K_DIR / metadata_file
    if not metadata_path.is_file():
        hf_hub_download(
            repo_id=HRP4K_REPO,
            repo_type="dataset",
            filename=metadata_file,
            local_dir=str(HRP4K_DIR),
        )
    with open(metadata_path) as file:
        rows = [json.loads(line) for line in file if line.strip()]
    selected = random.Random(SEED).sample(rows, HRP4K_IMAGES)
    image_files = [f"data/images/train/{row['file_name']}" for row in selected]
    missing_files = [filename for filename in image_files if not (HRP4K_DIR / filename).is_file()]
    if missing_files:
        print(f"Downloading {len(missing_files)} missing HRP4K images to {HRP4K_DIR}...")
        snapshot_download(
            repo_id=HRP4K_REPO,
            repo_type="dataset",
            allow_patterns=missing_files,
            local_dir=str(HRP4K_DIR),
        )
    else:
        print(f"Using {len(selected)} HRP4K images already downloaded in {HRP4K_DIR}.")
    return [
        (HRP4K_DIR / filename, row["objects"]["bbox"])
        for filename, row in zip(image_files, selected)
    ]


def split_local_samples(samples):
    """Share the exact local training/validation split with the viewer."""
    if len(samples) < 2:
        raise ValueError("At least two local images are needed for training and validation.")
    shuffled = list(samples)
    random.Random(SEED).shuffle(shuffled)
    valid_count = max(1, round(len(shuffled) * 0.2))
    return shuffled[valid_count:], shuffled[:valid_count]
