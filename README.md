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

The first run downloads the HRP4K subset into `data/hrp4k_original`, preserving the source layout (`data/images/train/` inside that folder). Later runs reuse local metadata and images, downloading only missing files. Prepared COCO splits go into `data/finetune_hrp4k`.

Each run saves logs and models in its own timestamped folder under `out/hrp4k-rfdetr-small-finetuned`, printed at startup. A full checkpoint is saved after every epoch (`checkpoint_0.ckpt`, `checkpoint_1.ckpt`, etc., with zero-based epoch numbers), including optimizer state for resuming. The best model is also saved as `checkpoint_best_total.pth`.

Training uses fixed **896 × 896** inputs, a tqdm progress bar, batch size 2, and 8 gradient accumulation steps. Edit the epoch count and other settings at the top of the training script. Dataset size and random seed are in `src/pothole_data.py`. The progress bar starts after model initialization and any initial CUDA kernel compilation.

The DINOv2 visual encoder is frozen, including its embeddings, transformer blocks, and normalization layers. The feature projector, detection transformer, object queries, and classification/box heads remain trainable. Set `freeze_encoder=False` in the model constructor to unfreeze the visual encoder.

To run inference with the fine-tuned checkpoint, replace `<run-timestamp>` with your run folder's name:

```bash
uv run scripts/inference_hrp4k_rfdetr_small.py \
  --weights "out/hrp4k-rfdetr-small-finetuned/<run-timestamp>/checkpoint_best_total.pth" \
  --output-dir out/finetuned-predictions
```

## Evaluate

Set `MODEL_DIR` at the top of `src/evaluate_rfdetr_small.py` to your timestamped training folder. The script finds **every `.pth` and `.ckpt` file** directly inside it and compares each one against the original HRP4K base model. Epoch checkpoints are evaluated in numerical order, followed by the best/EMA checkpoints.

To evaluate only selected checkpoints, set this list in the script:

```python
CHECKPOINTS = ["checkpoint_0.ckpt", "checkpoint_9.ckpt", "checkpoint_best_total.pth"]
```

Set `CHECKPOINTS = None` to evaluate all. A list is evaluated in the order provided; the base model is always included. Missing files are reported before evaluation starts.

Control evaluation size with `TRAIN_MY_DATASET_FRACTION`, `TRAIN_HRP_FRACTION`, `VAL_MY_DATASET_FRACTION`, and `VAL_HRP_FRACTION` in the same script: `1.0` uses all images; `0.25` uses roughly 25%, with at least one image. `VALIDATION_SEED` fixes the random subsets for all four splits so every checkpoint and the base model see the same images. HRP4K validation sampling happens before download, so only the selected validation images are fetched. Fractions, seed, and actual image counts are saved with the results. These settings affect evaluation only, not fine-tuning.

```bash
uv run src/evaluate_rfdetr_small.py
```

The script compares all models on the same four splits:

| Split | Images |
| --- | --- |
| `train_my_dataset` | Selected fraction of the prepared local training split |
| `val_my_dataset` | Selected fraction of the prepared local validation split |
| `train_hrp` | Selected fraction of the 2,000 prepared HRP4K training images |
| `val_hrp` | Selected fraction of HRP4K's official validation split |

It reads `data/finetune_hrp4k` without modifying it. Missing HRP4K validation images are downloaded into `data/hrp4k_original`; existing files are reused.

Reports **mAP50**, **mAP75**, **mAP50–95**, **mAR100**, **precision**, **recall**, and **F1**, all as percentages. For object detection, these replace a misleading single “accuracy” value. All models use 896 × 896 inputs and at most 100 pothole predictions per image. AP uses all prediction confidence levels; precision/recall/F1 use confidence ≥ 0.20 and IoU ≥ 0.50, with one-to-one box matching. Images without potholes still contribute false positives.

Results are saved **inside that model's folder**, under `evaluation/<timestamp>/`:

- `metrics.csv` and `results.json`: each checkpoint's scores on all four splits, including the base model.
- `curves.csv` and `curves.json`: precision/recall/F1 at confidence thresholds from 0 to 1.
- `splits.json`: the exact images and target boxes used.
- `plots/`: metric comparisons, precision–recall curves, confidence curves, detection counts, and training history.

The training-history graph reads the existing `metrics.csv` from the model folder (or Lightning log folders). Its validation scores retain RF-DETR's original threshold selection; the new evaluation uses the fixed thresholds above. The original training logs are preserved. Precision–recall curves are used for detection; a standard ROC curve needs true negatives, which bounding-box detection does not define.

To regenerate graphs from the latest saved evaluation without rerunning inference:

```bash
uv run src/evaluate_rfdetr_small.py --plots-only
```

Keep the prepared dataset used by the checkpoint you are evaluating; regenerating it with a different subset changes the comparison. Training scores measure fit to seen data. HRP4K validation was already used to select the base checkpoint, and local validation selects fine-tuned checkpoints, so these are validation comparisons rather than an independent final test.

## Run a saved model and view predictions

Set `WEIGHTS` at the top of `src/predict_and_view.py` to any saved checkpoint, then run:

```bash
uv run src/predict_and_view.py
```

This self-contained script predicts on every image in `data/processed_images`, saves annotated images and `predictions.json` beside the checkpoint in `predictions/<checkpoint>_<timestamp>/`, and starts **http://localhost:8000**. It includes both local training and validation images, without evaluation sampling.

Search filenames, filter images with/without detections, switch to originals, and click to enlarge. Use **← / →** to browse and **Esc** to close. Stop the viewer with **Ctrl+C**.

You can override `--weights`, `--input-dir`, `--confidence`, or `--port`. To reopen previously saved predictions without rerunning the model:

```bash
uv run src/predict_and_view.py --results-dir "out/hrp4k-rfdetr-small-finetuned/<run-timestamp>/predictions/<prediction-folder>"
```

## View dataset annotations

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
- `src/evaluate_rfdetr_small.py`: base versus fine-tuned evaluation.
- `src/plot_evaluation.py`: graphs from saved evaluation and training metrics.
- `src/predict_and_view.py`: self-contained saved-model inference and localhost viewer.
- `src/pothole_data.py`: shared dataset paths, selection, and loading.
- `scripts/Pothole_detection.ipynb`: original Colab notebook.

HRP4K was created by Hanshen Chen, Zhoulin Tu, Yu Zhao, and Jianfeng Ye ([paper](https://doi.org/10.1038/s41597-026-07317-w)). The [Hugging Face redistribution](https://huggingface.co/datasets/dronefreak/HRP4K) is licensed under CC BY 4.0.
