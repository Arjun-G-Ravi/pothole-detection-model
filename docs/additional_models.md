# D-FINE-S and DEIMv2-S

These scripts fine-tune the authors' actual Small architectures on the existing
`data/finetune_hrp4k` dataset used by RF-DETR and YOLO. No new random split or
HRP4K sample is generated. Every image entry, repeated entry, empty image, and
pothole box is preserved. Only the pothole category ID changes from 1 to 0 in
per-run annotation copies; the source dataset is never edited.

## One-time local setup

No GitHub account, token, or private repository access is needed. Download the
public source ZIPs in your browser and extract them to these directories
(the directories must directly contain `configs/` and the authors' code):

- [D-FINE ZIP](https://github.com/Peterande/D-FINE/archive/refs/heads/master.zip)
  → `out/vendor/D-FINE/`
- [DEIMv2 ZIP](https://github.com/Intellindust-AI-Lab/DEIMv2/archive/refs/heads/main.zip)
  → `out/vendor/DEIMv2/`

The [D-FINE](https://github.com/Peterande/D-FINE) and
[DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) implementations provide
their architectures, losses, trainers, and evaluators. They are not installed
by `uv sync` for this project. DEIMv2 documents Python 3.11 with PyTorch 2.5.1
and torchvision 0.20.1; these adapters use separate environments for both
models so the existing RF-DETR/YOLO dependencies can remain intact.

Run from this project root, once you have extracted the source:

```bash
uv venv --python 3.11 out/vendor/D-FINE/.venv
uv pip install --python out/vendor/D-FINE/.venv/bin/python -r src/detr_requirements.txt
uv venv --python 3.11 out/vendor/DEIMv2/.venv
uv pip install --python out/vendor/DEIMv2/.venv/bin/python -r src/detr_requirements.txt
```

These are setup commands for you to run. The scripts never install dependencies
or fetch source repositories. Paths and runtime executables are editable in
`src/dfine/config.py` and `src/deimv2/config.py`.

## Prepare or train

Check the dataset and generate annotations/configs without ML dependencies,
downloads, or starting training:

```bash
.venv/bin/python src/dfine/finetune_dfine_small.py --prepare-only
.venv/bin/python src/deimv2/finetune_deimv2_small.py --prepare-only
```

Start fine-tuning:

```bash
.venv/bin/python src/dfine/finetune_dfine_small.py
.venv/bin/python src/deimv2/finetune_deimv2_small.py
```

Each command runs one model in the foreground. The first training run downloads
public COCO-pretrained weights: `dfine_s_coco.pth` from the D-FINE authors, or
[`Intellindust/DEIMv2_DINOv3_S_COCO`](https://huggingface.co/Intellindust/DEIMv2_DINOv3_S_COCO).
DEIMv2's safetensors are converted to a cached PyTorch checkpoint and its
backbone layer selection is read from the matching published configuration.
No account token is used for the Hugging Face download. Use `--weights /path/to/model.pth`
to use an already downloaded official PyTorch checkpoint without fetching weights.

Defaults in each model's `config.py`:

| Setting | D-FINE-S | DEIMv2-S |
| --- | --- | --- |
| Epochs | 100 | 100 |
| Input size | 896 × 896 | 896 × 896 |
| Batch size | 32 | 32 |
| Optimizer | AdamW | AdamW |
| Learning rate | 0.0002 | 0.0005 |
| Backbone | Frozen | Frozen |
| Workers / seed | 2 / 42 | 2 / 42 |

CUDA is chosen when available, otherwise CPU. Override with `--device cpu` or
`--device cuda`. AMP is enabled on CUDA. Reduce `BATCH_SIZE` if GPU memory is
insufficient. Frozen backbones also stay in evaluation mode to preserve running
normalization statistics; set `FREEZE_BACKBONE=False` to train the whole model.
Compatible pretrained feature/box tensors are loaded, and incompatible COCO
class heads are initialized for the single pothole class.

The shared fine-tuning recipe uses horizontal flipping and photometric distortion,
fixed resolution, and cosine learning-rate decay with 100 warmup iterations.
DEIMv2 also applies the normalization required by its ViT backbone. Mosaic,
MixUp, CopyBlend, multiscale batching, and the upstream stage-restart schedule
are disabled. This is a controlled fine-tuning recipe, not a reproduction of
the papers' full COCO training schedules. Validation uses no random augmentation.

## Outputs and resuming

Runs are saved under `out/dfine-small-finetuned/<timestamp>/` and
`out/deimv2-small-finetuned/<timestamp>/`. They contain:

- `config.yaml`, `resolved_config.yaml`, and `run.json`: requested/resolved settings,
  runtime paths, source revision (when available), and dataset counts/hashes.
- `dataset/`: converted annotations and a source manifest; images stay in their original folders.
- `last.pth`, `checkpoint0000.pth`, etc.: resumable full-state checkpoints saved every epoch.
- `best_stg1.pth`: the upstream best validation checkpoint, when validation AP improves.
- `log.txt`: official per-epoch JSONL metrics, and `eval/`: COCO evaluation artifacts.
- `metrics.csv`: mAP50–95, mAP50, mAP75, and training loss, exported when training
  finishes or exits through a Python exception. AP values are fractions from 0 to 1.

Resume in the original run folder, preserving the optimizer and training progress:

```bash
.venv/bin/python src/dfine/finetune_dfine_small.py --resume out/dfine-small-finetuned/<timestamp>/last.pth
.venv/bin/python src/deimv2/finetune_deimv2_small.py --resume out/deimv2-small-finetuned/<timestamp>/last.pth
```

Resume uses the saved configuration and runtime rather than newly edited defaults.
Changed source annotation hashes are rejected. Keep the run folder and source
dataset available for inference and resuming.

## Compare predictions

To compare all four models, supply the additional saved checkpoints to the existing viewer:

```bash
.venv/bin/python src/compare_predictions.py \
  --dfine-weights out/dfine-small-finetuned/<timestamp>/best_stg1.pth \
  --deimv2-weights out/deimv2-small-finetuned/<timestamp>/best_stg1.pth
```

To compare only the two new models:

```bash
.venv/bin/python src/compare_predictions.py --models dfine deimv2 \
  --dfine-weights out/dfine-small-finetuned/<timestamp>/last.pth \
  --deimv2-weights out/deimv2-small-finetuned/<timestamp>/last.pth
```

Open **http://localhost:8000**. Training and Validation boxes, image search,
originals, confidence scores, and saved offline viewers work with any selected
models. Each new model predicts in its own runtime process; checkpoints must
sit beside the `run.json` and `resolved_config.yaml` produced by these scripts.
RF-DETR/YOLO-only saved comparisons remain compatible.
