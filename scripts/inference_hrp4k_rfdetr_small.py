"""Save annotated pothole predictions for the preprocessed images."""

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed_images",
        help="Folder containing preprocessed images (searched recursively).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "out" / "hrp4k-rfdetr-small",
        help="Folder for annotated images.",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.20, help="Detection threshold (default: %(default)s)."
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Inference device; auto selects CUDA when available, otherwise CPU.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="Local checkpoint; defaults to downloading the notebook's checkpoint.",
    )
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1.")
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")
    if args.output_dir == args.input_dir or args.input_dir in args.output_dir.parents:
        parser.error("Output directory must be outside the input directory.")
    if args.weights is not None:
        args.weights = args.weights.expanduser().resolve()
        if not args.weights.is_file():
            parser.error(f"Checkpoint does not exist: {args.weights}")
    return args


def main():
    args = parse_args()
    image_paths = sorted(
        path
        for path in args.input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise SystemExit(f"No supported images found in {args.input_dir}")

    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    from rfdetr import RFDETRSmall
    from tqdm import tqdm

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Input: {args.input_dir} ({len(image_paths)} images)")
    print(f"Output: {args.output_dir}")
    print(f"Device: {device}")
    weights = str(args.weights) if args.weights else hf_hub_download(
        repo_id="dronefreak/hrp4k-rfdetr-small",
        filename="checkpoint_best_total.pth",
    )
    model = RFDETRSmall(pretrain_weights=weights, device=device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=24)
    total_detections = 0
    saved = 0
    failed = 0

    for path in tqdm(image_paths, desc="Detecting potholes", unit="image"):
        try:
            with Image.open(path) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")

            with torch.inference_mode():
                detections = model.predict(image, threshold=args.confidence)

            draw = ImageDraw.Draw(image)
            line_width = max(3, min(image.size) // 400)
            for box, confidence in zip(detections.xyxy, detections.confidence):
                x1, y1, x2, y2 = map(int, box)
                draw.rectangle([x1, y1, x2, y2], outline="red", width=line_width)
                label = f"Pothole {confidence:.2f}"
                left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
                text_width, text_height = right - left, bottom - top
                label_y = max(0, y1 - text_height - 10)
                draw.rectangle(
                    [x1, label_y, x1 + text_width + 12, label_y + text_height + 10],
                    fill="red",
                )
                draw.text(
                    (x1 + 6 - left, label_y + 5 - top),
                    label,
                    fill="white",
                    font=font,
                )

            # Keep filenames and subfolders, including images with no detections.
            output_path = args.output_dir / path.relative_to(args.input_dir)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, quality=95)
            total_detections += len(detections.xyxy)
            saved += 1
        except Exception as exc:
            failed += 1
            tqdm.write(f"Error processing {path}: {exc}")

    print(f"Finished: {saved} images saved, {total_detections} pothole boxes, {failed} failures.")
    print(f"Results saved to: {args.output_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
