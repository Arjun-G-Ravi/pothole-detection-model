"""Fine-tune YOLO26 Small on the existing RF-DETR dataset; edit config.py."""

from datetime import datetime

from config import OUTPUT_DIR, TRAIN_SETTINGS, WEIGHTS
from prepare_dataset import prepare_dataset


def main():
    import torch
    from ultralytics import YOLO, settings

    dataset_yaml = prepare_dataset()
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Run directory: {OUTPUT_DIR / stamp}\nDevice: {device}")
    settings.update({"wandb": False})
    model = YOLO(str(WEIGHTS))
    model.train(
        data=str(dataset_yaml),
        project=str(OUTPUT_DIR),
        name=stamp,
        device=device,
        **TRAIN_SETTINGS,
    )
    print(f"Fine-tuned checkpoints saved to: {OUTPUT_DIR / stamp / 'weights'}")


if __name__ == "__main__":
    main()
