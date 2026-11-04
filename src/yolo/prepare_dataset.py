"""Convert the existing RF-DETR COCO splits to YOLO labels without downloads."""

import json
import math
from collections import defaultdict

from config import DATASET_DIR, SOURCE_DATASET


def yolo_box(box, width, height):
    """Convert a COCO pixel box to normalized YOLO center/size coordinates."""
    x, y, w, h = box
    tolerance = 1e-6  # Pixel rounding from COCO coordinate conversions.
    if (not all(math.isfinite(value) for value in (*box, width, height))
            or width <= 0 or height <= 0 or w <= 0 or h <= 0
            or x < -tolerance or y < -tolerance
            or x + w > width + tolerance or y + h > height + tolerance):
        raise ValueError(f"Invalid COCO box {box} for image size {width}x{height}")
    right, bottom = min(width, x + w), min(height, y + h)
    x, y = max(0, x), max(0, y)
    w, h = right - x, bottom - y
    return (x + w / 2) / width, (y + h / 2) / height, w / width, h / height


def prepare_dataset(source=SOURCE_DATASET, destination=DATASET_DIR):
    """Preserve entries and split membership; only annotation format changes."""
    source, destination = source.resolve(), destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and YOLO dataset folders must be separate.")
    counts = {}
    for source_split, split in (("train", "train"), ("valid", "val")):
        folder = source / source_split
        coco = json.loads((folder / "_annotations.coco.json").read_text())
        pothole_ids = {category["id"] for category in coco["categories"]
                       if category["name"].lower() == "pothole"}
        if not pothole_ids:
            raise ValueError(f"No pothole category in {folder}")
        boxes = defaultdict(list)
        for annotation in coco["annotations"]:
            if annotation["category_id"] in pothole_ids:
                boxes[annotation["image_id"]].append(annotation["bbox"])
        images_dir, labels_dir = destination / "images" / split, destination / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for image in coco["images"]:
            original = folder / image["file_name"]
            if not original.is_file():
                raise FileNotFoundError(f"Prepared RF-DETR image not found: {original}")
            # COCO IDs keep repeated images and equal filename stems distinct.
            name = f"{image['id']:05d}{original.suffix.lower()}"
            linked = images_dir / name
            if linked.is_symlink():
                linked.unlink()
            linked.symlink_to(original)
            lines = ["0 " + " ".join(f"{value:.10f}" for value in
                     yolo_box(box, image["width"], image["height"]))
                     for box in boxes[image["id"]]]
            (labels_dir / f"{linked.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            entries.append(str(linked))
        # Lists ensure files left from an earlier conversion are never included.
        (destination / f"{split}.txt").write_text("\n".join(entries) + "\n")
        (destination / "labels" / f"{split}.cache").unlink(missing_ok=True)
        counts[split] = len(entries)
    dataset_yaml = destination / "dataset.yaml"
    dataset_yaml.write_text(
        f"path: {json.dumps(str(destination))}\n"
        "train: train.txt\nval: val.txt\nnames:\n  0: pothole\n"
    )
    print(f"YOLO dataset: {dataset_yaml} ({counts['train']} train, {counts['val']} validation)")
    return dataset_yaml


if __name__ == "__main__":
    prepare_dataset()
