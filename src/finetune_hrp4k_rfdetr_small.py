"""Fine-tune RF-DETR Small on local images and a subset of HRP4K."""

import json
import shutil

from pothole_data import (
    DATA_DIR,
    ROOT,
    SEED,
    load_hrp4k_samples,
    load_local_samples,
    split_local_samples,
)


DATASET_DIR = DATA_DIR / "finetune_hrp4k"
OUTPUT_DIR = ROOT / "out" / "hrp4k-rfdetr-small-finetuned"
MODEL_REPO = "dronefreak/hrp4k-rfdetr-small"
EPOCHS = 30
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 1e-5
ENCODER_LEARNING_RATE = 1e-6


def write_coco_split(split, samples):
    """Copy images into an RF-DETR split and give all pothole boxes class ID 1."""
    from PIL import Image
    from tqdm import tqdm

    split_dir = DATASET_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "pothole", "supercategory": "road-damage"}],
    }
    for image_id, (path, boxes) in enumerate(tqdm(samples, desc=f"Preparing {split}"), 1):
        filename = f"{image_id:05d}_{path.name}"
        with Image.open(path) as image:
            width, height = image.size
        shutil.copy2(path, split_dir / filename)
        coco["images"].append({
            "id": image_id, "file_name": filename, "width": width, "height": height,
        })
        for x, y, box_width, box_height in boxes:
            coco["annotations"].append({
                "id": len(coco["annotations"]) + 1,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x, y, box_width, box_height],
                "area": box_width * box_height,
                "iscrowd": 0,
            })
    (split_dir / "_annotations.coco.json").write_text(json.dumps(coco))


def main():
    import torch
    from huggingface_hub import hf_hub_download
    from rfdetr import RFDETRSmall

    local_train, valid_samples = split_local_samples(load_local_samples())
    hrp4k_samples = load_hrp4k_samples()

    write_coco_split("train", local_train + hrp4k_samples)
    write_coco_split("valid", valid_samples)
    print(f"Training: {len(local_train)} local + {len(hrp4k_samples)} HRP4K images")
    print(f"Validation: {len(valid_samples)} local images")

    weights = hf_hub_download(repo_id=MODEL_REPO, filename="checkpoint_best_total.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Restore the checkpoint's architecture and resolution along with its weights.
    model = RFDETRSmall.from_checkpoint(weights, device=device)
    model.train(
        dataset_dir=str(DATASET_DIR),
        output_dir=str(OUTPUT_DIR),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        lr=LEARNING_RATE,
        lr_encoder=ENCODER_LEARNING_RATE,
        device=device,
        num_workers=2,
        seed=SEED,
        tensorboard=False,
        wandb=False,
        run_test=False,
    )
    print(f"Fine-tuned checkpoints saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
