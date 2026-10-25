# Pothole detection

Run and fine-tune [RF-DETR Small](https://huggingface.co/dronefreak/hrp4k-rfdetr-small) locally, and browse images with pothole boxes.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from the project root:

```bash
uv sync
```

Dependencies, including training support, are listed in `pyproject.toml`. `uv.lock` pins their versions; `.python-version` selects Python 3.13.

## Detect potholes

```bash
uv run scripts/inference_hrp4k_rfdetr_small.py
```

Reads `data/processed_images` and saves annotated images to `out/hrp4k-rfdetr-small`. The confidence threshold defaults to `0.20`. CUDA is selected when available; otherwise it uses CPU.

Options: `--input-dir`, `--output-dir`, `--confidence`, `--device`, and `--weights`.

## Fine-tune

```bash
uv run src/finetune_hrp4k_rfdetr_small.py
```

Continues from the HRP4K checkpoint using:

- Training: 80% of your preprocessed images plus 2,000 HRP4K training images.
- Validation: the remaining 20% of your images.

Local labels come from `data/annotations/original_coco.json`. With the current dataset, this is 163 local training images and 41 validation images.

The first run downloads the HRP4K subset into `data/hrp4k_original`, preserving the source layout (`data/images/train/` inside that folder). Prepared COCO splits go into `data/finetune_hrp4k`; checkpoints go into `out/hrp4k-rfdetr-small-finetuned`.

Training defaults to 30 epochs, batch size 2, and 8 gradient accumulation steps. Edit the settings at the top of the training script. Dataset size and random seed are in `src/pothole_data.py`.

To run inference with the fine-tuned checkpoint:

```bash
uv run scripts/inference_hrp4k_rfdetr_small.py \
  --weights out/hrp4k-rfdetr-small-finetuned/checkpoint_best_total.pth \
  --output-dir out/finetuned-predictions
```

## View images

```bash
uv run scripts/view_potholes.py
```

Open **http://localhost:8000**. The viewer includes your images and the same 2,000 HRP4K images selected for training; the first run downloads them if needed.

Filter by dataset or training split, search filenames, and switch between annotations, saved predictions, and original images. Click an image to enlarge it; use **← / →** to browse and **Esc** to close. Stop the server with **Ctrl+C**.

Saved predictions default to `out/hrp4k-rfdetr-small`. To view fine-tuned predictions:

```bash
uv run scripts/view_potholes.py --predictions-dir out/finetuned-predictions
```

Use `--port` to change the server port. Saved predictions are available for local images that have inference output; HRP4K images display their dataset annotations.

## Files

- `scripts/inference_hrp4k_rfdetr_small.py`: inference and annotated image output.
- `scripts/view_potholes.py`: localhost viewer.
- `src/finetune_hrp4k_rfdetr_small.py`: dataset preparation and training.
- `src/pothole_data.py`: shared dataset paths, selection, and loading.
- `scripts/Pothole_detection.ipynb`: original Colab notebook.

HRP4K was created by Hanshen Chen, Zhoulin Tu, Yu Zhao, and Jianfeng Ye ([paper](https://doi.org/10.1038/s41597-026-07317-w)). The [Hugging Face redistribution](https://huggingface.co/datasets/dronefreak/HRP4K) is licensed under CC BY 4.0.
