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
uv run src/rfdert/finetune_hrp4k_rfdetr_small.py
```

Continues from the HRP4K checkpoint using:

- Training: 80% of your preprocessed images, repeated evenly to give 2,000 local training entries, plus 2,000 HRP4K training images (a 50/50 mix per epoch).
- Validation: the remaining 20% of your images.

Local labels come from `data/annotations/original_coco.json`, converted from `final_pothole_boxes.json` (203 images and 571 pothole boxes, including one image without potholes). With the current dataset, this is 162 local training images and 41 validation images. COCO boxes use `[x, y, width, height]` and reference `data/processed_images`. The old augmented annotations have been removed.

Each local training image appears 12 or 13 times per epoch; seed 42 determines the extra repeats. Validation images are never repeated or included in training. These are repeated training examples, not additional unique images. Prepared training-set evaluation includes those repetitions.

The first run downloads the HRP4K subset into `data/hrp4k_original`, preserving the source layout (`data/images/train/` inside that folder). Later runs reuse local metadata and images, downloading only missing files. Prepared COCO splits go into `data/finetune_hrp4k`.

Each run saves logs and models in its own timestamped folder under `out/hrp4k-rfdetr-small-finetuned`, printed at startup. A full checkpoint is saved after every epoch (`checkpoint_0.ckpt`, `checkpoint_1.ckpt`, etc., with zero-based epoch numbers), including optimizer state for resuming. The best model is also saved as `checkpoint_best_total.pth`.

Training runs for **100 epochs**, using fixed **896 × 896** inputs, a tqdm progress bar, batch size 32, and one gradient accumulation step. Edit the epoch count and other settings at the top of the training script. Dataset size and random seed are in `src/pothole_data.py`. The progress bar starts after model initialization and any initial CUDA kernel compilation.

The DINOv2 visual encoder is frozen, including its embeddings, transformer blocks, and normalization layers. The feature projector, detection transformer, object queries, and classification/box heads remain trainable. Set `freeze_encoder=False` in the model constructor to unfreeze the visual encoder.

To run inference with the fine-tuned checkpoint, replace `<run-timestamp>` with your run folder's name:

```bash
uv run scripts/inference_hrp4k_rfdetr_small.py \
  --weights "out/hrp4k-rfdetr-small-finetuned/<run-timestamp>/checkpoint_best_total.pth" \
  --output-dir out/finetuned-predictions
```

## Fine-tune YOLO26 Small

YOLO uses the exact prepared RF-DETR splits, including the repeated local
training images: 4,000 training entries and 41 validation images with the current
dataset. Prepare the RF-DETR dataset first if `data/finetune_hrp4k` is missing.
YOLO preparation reads these local files without downloading or changing them.

Install the project's dependencies yourself, including Ultralytics:

```fish
uv sync --locked
```

Edit `src/yolo/config.py` to change weights, paths, or training settings. The
defaults use official `yolo26s.pt` general pretrained weights, freeze backbone
modules 0–10, and train with AdamW at `1e-3`, batch size 32, 896-pixel inputs,
and up to 100 epochs. Early stopping uses patience 10. The nominal batch size
is 32, so there is one optimizer update per batch. The learning rate decays
linearly to 1% of its initial value, with three warmup epochs; other augmentation
and precision settings use Ultralytics defaults. CUDA is used when available,
otherwise CPU.

Run from the project root without command-line options:

```fish
.venv/bin/python src/yolo/finetune_yolo26_small.py
```

The script prepares `data/finetune_yolo26` and starts training with local metrics,
plots, and checkpoints. W&B is disabled and requires no login. The first run may
download pretrained model weights. Dataset images are linked locally, and COCO
annotations are converted to normalized YOLO labels with class `0` for potholes.
Empty images keep empty label files. Train/validation lists preserve every
prepared entry and ignore stale images from earlier conversions.

To prepare and check the dataset without starting training or downloading weights:

```fish
.venv/bin/python src/yolo/prepare_dataset.py
```

Each run saves resolved training arguments (`args.yaml`), metrics (`results.csv`),
plots, and checkpoints under `out/yolo26-small-finetuned/<timestamp>/`.
The `weights/` folder contains `best.pt`, `last.pt`, and epoch checkpoints.
YOLO starts from general pretrained weights; RF-DETR starts from pothole-trained
weights, so they have different starting points despite using identical data.
Separate YOLO evaluation is not included. Use `src/compare_predictions.py` below
to compare its predictions with RF-DETR.

## Evaluate

Set `MODEL_DIR` at the top of `src/rfdert/evaluate_rfdetr_small.py` to your timestamped training folder. The script finds **every `.pth` and `.ckpt` file** directly inside it and compares each one against the original HRP4K base model. Epoch checkpoints are evaluated in numerical order, followed by the best/EMA checkpoints.

To evaluate only selected checkpoints, set this list in the script:

```python
CHECKPOINTS = ["checkpoint_0.ckpt", "checkpoint_9.ckpt", "checkpoint_best_total.pth"]
```

Set `CHECKPOINTS = None` to evaluate all. A list is evaluated in the order provided; the base model is always included. Missing files are reported before evaluation starts.

Control evaluation size with `TRAIN_MY_DATASET_FRACTION`, `TRAIN_HRP_FRACTION`, `VAL_MY_DATASET_FRACTION`, and `VAL_HRP_FRACTION` in the same script: `1.0` uses all images; `0.25` uses roughly 25%, with at least one image. `VALIDATION_SEED` fixes the random subsets for all four splits so every checkpoint and the base model see the same images. HRP4K validation sampling happens before download, so only the selected validation images are fetched. Fractions, seed, and actual image counts are saved with the results. These settings affect evaluation only, not fine-tuning.

```bash
uv run src/rfdert/evaluate_rfdetr_small.py
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
uv run src/rfdert/evaluate_rfdetr_small.py --plots-only
```

Keep the prepared dataset used by the checkpoint you are evaluating; regenerating it with a different subset changes the comparison. Training scores measure fit to seen data. HRP4K validation was already used to select the base checkpoint, and local validation selects fine-tuned checkpoints, so these are validation comparisons rather than an independent final test.

## One-time graphs for the 25-epoch run

```bash
uv run scripts/plot_25_epoch_run.py
```

Saves graphs and their data to `scripts/graphs`: F1/precision/recall and mAP over epochs, losses, learning rates, checkpoint comparisons, precision–recall and confidence curves, FROC, and an image-level ROC comparing the base and best models. ROC inference uses the saved HRP4K validation subset and caches its predictions for later reruns. The ROC measures pothole presence per image; the PR and FROC graphs measure box detection.

## Fine-tune D-FINE-S and DEIMv2-S

Both models reuse the exact prepared RF-DETR/YOLO training and validation data.
See [setup, configuration, checkpoints, and prediction comparison](docs/additional_models.md)
for their separate official runtimes. After that one-time setup, run:

```bash
.venv/bin/python src/dfine/finetune_dfine_small.py
.venv/bin/python src/deimv2/finetune_deimv2_small.py
```

Add `--prepare-only` to validate data and write configs without installing or
running the models. Defaults are 100 epochs, 896-pixel inputs, batch size 32,
and frozen backbones, with timestamped outputs and a checkpoint every epoch.

## Compare all four models

```fish
/home/arjun/ai_env/bin/python src/compare_predictions.py
```

Runs the saved RF-DETR Small, YOLO26 Small, D-FINE-S, and DEIMv2-S models on
every image in `data/processed_images` and starts **http://localhost:8000**.
Each image displays the four predictions in a two-column grid, with numbered
pothole boxes, confidence scores, and box coordinates.
Separate **Training data** and **Validation data** boxes show image counts;
click a box to browse only that split. Membership comes from the prepared
training dataset, or the shared seeded training split if it has not been prepared.
Only images belonging to those splits appear in the viewer.
Use Previous/Next or **← / →** to browse, select an image by filename, search,
filter by detection counts, or switch to the original. Click any image to
open it at full size. Stop the server with **Ctrl+C**.

Edit `RFDETR_WEIGHTS`, `YOLO_WEIGHTS`, `DFINE_WEIGHTS`, and `DEIMV2_WEIGHTS`
at the top of the script to choose checkpoints. The defaults use the saved best
RF-DETR EMA checkpoint from November 2, YOLO checkpoint from November 3, and
D-FINE/DEIMv2 checkpoints from November 4. Input paths, confidence, resolution,
and port are also configurable at the top of the script.

All models receive the same original images, a confidence threshold of 0.30,
and an input size of 896. Only predictions labeled `pothole` are shown; images
without predictions remain browsable. Models load one at a time to limit GPU
memory use. CUDA is selected when available, otherwise CPU; use `--device` to
override it. This is a visual comparison on all local images, including training
images, rather than an evaluation on an independent test set.

Originals, annotated images, `predictions.json`, and a standalone `index.html`
are saved in a new timestamped folder under `out/model-comparison`. Use
`--input-dir`, `--output-dir`, or `--port` to change those settings. Add
`--no-serve` to save results without starting a server; open the saved
`index.html` directly in a browser, or reopen it later without running inference:

```bash
uv run src/compare_predictions.py --results-dir out/model-comparison/<timestamp>
```

Reopening saved results also updates the viewer and adds missing split labels
without rerunning inference.

All four models are included by default. Use `--models dfine deimv2` to compare
only those models; see [examples](docs/additional_models.md#compare-predictions).

## Run a saved model and view predictions

Set `WEIGHTS` at the top of `src/rfdert/predict_and_view.py` to any saved checkpoint, then run:

```bash
uv run src/rfdert/predict_and_view.py
```

This self-contained script predicts on every image in `data/processed_images`, saves annotated images and `predictions.json` beside the checkpoint in `predictions/<checkpoint>_<timestamp>/`, and starts **http://localhost:8000**. It includes both local training and validation images, without evaluation sampling.

Search filenames, filter images with/without detections, switch to originals, and click to enlarge. Use **← / →** to browse and **Esc** to close. Stop the viewer with **Ctrl+C**.

You can override `--weights`, `--input-dir`, `--confidence`, or `--port`. To reopen previously saved predictions without rerunning the model:

```bash
uv run src/rfdert/predict_and_view.py --results-dir "out/hrp4k-rfdetr-small-finetuned/<run-timestamp>/predictions/<prediction-folder>"
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
- `src/rfdert/finetune_hrp4k_rfdetr_small.py`: dataset preparation and training.
- `src/rfdert/evaluate_rfdetr_small.py`: base versus fine-tuned evaluation.
- `src/rfdert/plot_evaluation.py`: graphs from saved evaluation and training metrics.
- `src/rfdert/predict_and_view.py`: self-contained saved-model inference and localhost viewer.
- `src/compare_predictions.py`: four-model prediction comparison and standalone saved viewer.
- `src/pothole_data.py`: shared dataset paths, selection, and loading.
- `src/yolo/config.py`: YOLO26 training configuration.
- `src/yolo/prepare_dataset.py`: local COCO-to-YOLO conversion.
- `src/yolo/finetune_yolo26_small.py`: YOLO26 Small fine-tuning with local logs.
- `scripts/Pothole_detection.ipynb`: original Colab notebook.

HRP4K was created by Hanshen Chen, Zhoulin Tu, Yu Zhao, and Jianfeng Ye ([paper](https://doi.org/10.1038/s41597-026-07317-w)). The [Hugging Face redistribution](https://huggingface.co/datasets/dronefreak/HRP4K) is licensed under CC BY 4.0.
