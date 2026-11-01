    # Pothole Detection: Fine-tuning and Complete Project Guide

    Project documentation • 28 October 2026

    ## 1. Project purpose
    This project detects potholes in road images using an HRP4K-pretrained RF-DETR Small model. It supports batch prediction, fine-tuning on local images mixed with HRP4K, checkpoint evaluation, graphs, and browser-based inspection. Outputs are rectangular bounding boxes and confidence scores. Segmentation, road severity estimation, video tracking, and a deployed web service are not implemented.

    ## 2. Setup
    Run commands from the repository root. Python 3.13 is selected by .python-version. Install uv, then synchronize the environment:

    ```bash
    uv sync
    ```

    pyproject.toml declares RF-DETR with training support, PyTorch, torchvision, torchmetrics detection, Hugging Face Hub, Pillow, and tqdm. uv.lock pins the environment. CUDA is selected when available; training and evaluation otherwise fall back to CPU. Initial use requires access to Hugging Face for model weights and missing dataset images.

    ## 3. Dataset and preparation
    The inspected local dataset contains 204 processed images and 258 pothole annotations. Labels are read from data/annotations/original_coco.json; image paths are resolved relative to data/. Boxes use COCO [x, y, width, height] pixel coordinates. All 204 annotated local images contain at least one pothole.

    src/pothole_data.py shuffles local images using seed 42 and holds out 20%: 163 local training images with 205 boxes and 41 validation images with 53 boxes. It also selects 2,000 HRP4K training images with the same seed. HRP4K metadata and images are cached under data/hrp4k_original/data/images/train/. Subsequent runs download only missing files.

    The combined prepared training split contains 2,163 images and 2,663 boxes; 696 training images have no pothole annotations. Validation contains only the 41 local images. The training preparation copies images to data/finetune_hrp4k/train/ and valid/, prefixes filenames with sequential IDs, and writes _annotations.coco.json in each split. Pothole category ID is normalized to 1. Image dimensions are read from the source files.

    Existing data/augmented_images and augmented_coco.json are not used by the active training loader. The inspected augmented annotation file contains 1,632 image records but only 8 annotations, so it should not be substituted for the original annotations without review. The repository consumes existing processed images; its active scripts do not recreate the original image preprocessing pipeline.

    ## 4. Fine-tuning process
    The entry point is src/finetune_hrp4k_rfdetr_small.py:

    ```bash
    uv run src/finetune_hrp4k_rfdetr_small.py
    ```

    The script creates a timestamped run directory, loads and splits local data, fetches the seeded HRP4K subset, writes prepared COCO splits, and downloads checkpoint_best_total.pth from dronefreak/hrp4k-rfdetr-small. It restores the model using RFDETRSmall.from_checkpoint with explicit 896-pixel resolution because the base checkpoint lacks its saved resolution.

    The DINOv2 visual encoder is frozen. The feature projector, detection transformer, object queries, and classification/box heads remain trainable. Set freeze_encoder=False in the constructor to unfreeze the encoder for a separate experiment.

    Current training settings, confirmed by the saved 25-epoch run configuration:

    | Setting | Value |
    | --- | --- |
    | Model | RF-DETR Small, one pothole class |
    | Input resolution | 896 × 896 |
    | Epochs | 25 |
    | Batch size | 4 |
    | Gradient accumulation | 8 steps |
    | Nominal effective batch, one device | 32 images |
    | Learning rate | 0.00001 |
    | Encoder learning rate | 0.000001; encoder remains frozen |
    | Multi-scale training | Disabled |
    | Data-loader workers | 2 |
    | Seed | 42 |
    | Checkpoint interval | Every epoch |
    | TensorBoard / W&B / final test | Disabled |

    Edit these constants at the top of the training script before starting another run. The README mentions batch size 2, but the current script and recorded run use 4. The recorded optimizer is AdamW; the saved configuration enables EMA and selects best models using mAP. The encoder learning-rate setting does not update frozen encoder parameters.

    The prepared dataset location is shared across runs, while checkpoints and logs are separated by timestamp. Preserve the prepared split used for each experiment: rerunning preparation with changed data or selection settings can invalidate later comparisons.

    ## 5. Checkpoints and training logs
    Each run is written under out/hrp4k-rfdetr-small-finetuned/<run-timestamp>/. Full epoch checkpoints are named checkpoint_0.ckpt through checkpoint_24.ckpt for a 25-epoch run. Filenames are zero-based: checkpoint_24.ckpt represents completed epoch 25. These checkpoints contain training state for resumption, while checkpoint_best_total.pth is a selected model artifact.

    training_config.json records model and training configuration; metrics.csv records losses and validation metrics. The current training script starts a new run and does not expose a resume command-line option. Resumption requires explicitly configuring the underlying training call; rerunning the script alone starts fresh.

    ## 6. Batch inference
    For the public base checkpoint:

    ```bash
    uv run scripts/inference_hrp4k_rfdetr_small.py
    ```

    Input defaults to data/processed_images; annotated images are saved to out/hrp4k-rfdetr-small with matching relative paths. Confidence defaults to 0.20. Options are --input-dir, --output-dir, --confidence, --device, and --weights. Supported extensions are JPG, JPEG, PNG, WEBP, and BMP. The script searches recursively and reports image failures.

    For saved fine-tuned checkpoints, the dedicated viewer script restores checkpoint configuration explicitly and uses 896 × 896 inputs:

    ```bash
    uv run src/predict_and_view.py --weights "out/hrp4k-rfdetr-small-finetuned/2026-10-25_19-56-17_610318/checkpoint_best_total.pth"
    ```

    Its current confidence default is 0.30, which differs from the evaluator's 0.20. Override --confidence when comparable operating thresholds are needed. Prediction outputs go beside the checkpoint under predictions/<checkpoint>_<timestamp>/. The folder contains annotated images and predictions.json with source paths, resolution, confidence, per-image box coordinates in xyxy format, scores, and counts. The latest inspected saved prediction run processed 204 images and produced 331 boxes at confidence 0.30; this count is not an accuracy measure.

    ## 7. Evaluation procedure
    Edit MODEL_DIR at the top of src/evaluate_rfdetr_small.py to select a training run. CHECKPOINTS currently selects checkpoint_10.ckpt, checkpoint_20.ckpt, checkpoint_24.ckpt, and checkpoint_best_total.pth; the base model is always included. Set CHECKPOINTS=None to discover all .pth and .ckpt files directly inside that run folder.

    ```bash
    uv run src/evaluate_rfdetr_small.py
    ```

    Every model sees the same deterministic images. Current fractions are 100% of local training, 100% of local validation, 10% of the prepared HRP4K training subset, and 15% of official HRP4K validation, with seed 42. This gives 163, 41, 200, and 135 images respectively. HRP4K validation selection happens before downloading missing images.

    Evaluation predicts at confidence 0.0, filters the pothole class, sorts scores, and retains at most 100 detections per image. COCO targets are converted from xywh to xyxy. Precision, recall, and F1 use confidence ≥ 0.20, IoU ≥ 0.50, and confidence-ordered one-to-one matching. An unmatched prediction is a false positive; an unmatched target is a false negative. Empty target images still contribute false positives.

    Precision = TP / (TP + FP). Recall = TP / (TP + FN). F1 = 2 × precision × recall / (precision + recall). IoU measures box intersection divided by union. mAP50 and mAP75 measure average precision at IoU 0.50 and 0.75. mAP50–95 averages across COCO IoU thresholds; mAR100 measures average recall with a maximum of 100 predictions. AP uses the score ranking rather than the display threshold. Reported metrics are percentages.

    ## 8. Saved experimental results
    The results below come from run 2026-10-25_19-56-17_610318, evaluation 2026-10-26_07-44-52_888842. Values are percentages and use the evaluation settings above. Training scores measure performance on seen data.

    ### train_my_dataset

    | Model | mAP50 | mAP50–95 | Precision | Recall | F1 |
    | --- | --- | --- | --- | --- | --- |
    | base | 4.21 | 1.29 | 6.41 | 17.07 | 9.32 |
    | checkpoint_10.ckpt | 13.35 | 5.80 | 14.72 | 42.93 | 21.92 |
    | checkpoint_20.ckpt | 19.50 | 8.59 | 16.01 | 45.85 | 23.74 |
    | checkpoint_24.ckpt | 22.17 | 9.31 | 17.44 | 49.76 | 25.82 |
    | checkpoint_best_total.pth | 19.46 | 8.36 | 15.79 | 48.29 | 23.80 |

    ### val_my_dataset

    | Model | mAP50 | mAP50–95 | Precision | Recall | F1 |
    | --- | --- | --- | --- | --- | --- |
    | base | 2.14 | 0.69 | 2.84 | 9.43 | 4.37 |
    | checkpoint_10.ckpt | 12.03 | 6.74 | 12.72 | 41.51 | 19.47 |
    | checkpoint_20.ckpt | 16.84 | 10.27 | 13.77 | 43.40 | 20.91 |
    | checkpoint_24.ckpt | 17.86 | 10.81 | 15.29 | 45.28 | 22.86 |
    | checkpoint_best_total.pth | 16.51 | 9.90 | 12.17 | 43.40 | 19.01 |

    ### train_hrp

    | Model | mAP50 | mAP50–95 | Precision | Recall | F1 |
    | --- | --- | --- | --- | --- | --- |
    | base | 74.66 | 45.14 | 48.74 | 78.46 | 60.12 |
    | checkpoint_10.ckpt | 75.64 | 46.44 | 46.82 | 80.89 | 59.31 |
    | checkpoint_20.ckpt | 76.25 | 46.56 | 50.38 | 80.49 | 61.97 |
    | checkpoint_24.ckpt | 76.97 | 47.34 | 48.10 | 82.52 | 60.78 |
    | checkpoint_best_total.pth | 76.53 | 47.12 | 49.03 | 82.11 | 61.40 |

    ### val_hrp

    | Model | mAP50 | mAP50–95 | Precision | Recall | F1 |
    | --- | --- | --- | --- | --- | --- |
    | base | 52.51 | 26.57 | 39.73 | 58.55 | 47.34 |
    | checkpoint_10.ckpt | 53.61 | 26.78 | 40.17 | 61.84 | 48.70 |
    | checkpoint_20.ckpt | 53.24 | 26.92 | 43.06 | 59.21 | 49.86 |
    | checkpoint_24.ckpt | 52.94 | 26.95 | 41.89 | 61.18 | 49.73 |
    | checkpoint_best_total.pth | 53.33 | 27.23 | 42.13 | 59.87 | 49.46 |

    On local validation, epoch 25 improves mAP50 from 2.14% to 17.86% and F1 from 4.37% to 22.86%. Its counts are TP=24, FP=133, FN=29. Substantial false detections and missed potholes remain. The selected best_total artifact achieves local validation F1 19.01%; its name does not imply it leads every metric under this separate evaluation protocol.

    The best logged training-validation F1 is 33.33% at displayed epoch 23. Training logs retain RF-DETR's own threshold selection and evaluation settings; this value is not directly comparable to the evaluator's fixed-threshold F1. On HRP4K validation, the base and best_total models achieve F1 47.34% and 49.46% respectively.

    ## 9. Graphs and interpretation
    scripts/graphs contains F1, precision/recall, mAP, loss, learning-rate, checkpoint-comparison, precision–recall, confidence, FROC, and image-level ROC graphs, with exported CSV/JSON data.

    ```bash
    uv run scripts/plot_25_epoch_run.py
    uv run src/evaluate_rfdetr_small.py --plots-only
    ```

    The first command targets the saved 25-epoch run and may run inference for uncached ROC scores. It selects a complete saved evaluation and caches image scores. The second regenerates evaluation plots from the latest saved evaluation under MODEL_DIR without rerunning model inference.

    Detection PR curves show precision versus recall. FROC shows localization recall against false positives per image. Image-level ROC uses the maximum pothole score to classify whether an image contains any pothole; it does not measure box localization. The saved ROC subset has 85 positive and 50 negative images. AUC is 0.9087 for the base and 0.9099 for best_total; this small observed difference does not establish a statistically reliable improvement.

    ## 10. Browser viewers
    src/predict_and_view.py starts http://localhost:8000 after saving predictions. It supports filename search, detection/no-detection filters, originals, enlarged images, arrow-key navigation, and Escape to close. Use --port to change ports and Ctrl+C to stop.

    ```bash
    uv run src/predict_and_view.py --results-dir "<saved-prediction-folder>"
    uv run scripts/view_potholes.py
    uv run scripts/view_potholes.py --predictions-dir out/finetuned-predictions --port 8001
    ```

    --results-dir reopens predictions without inference. Original-image viewing still needs the paths recorded in predictions.json. The annotation viewer loads local images and the same 2,000 HRP4K training images, supports dataset/split filters, and displays annotations or saved local prediction images. Its first run may download missing HRP4K files. Both servers bind to 127.0.0.1.

    ## 11. Output and source map
    | Path | Purpose |
    | --- | --- |
    | src/pothole_data.py | Dataset paths, local labels, seeded split, HRP4K selection/download |
    | src/finetune_hrp4k_rfdetr_small.py | COCO preparation and training |
    | src/evaluate_rfdetr_small.py | Base/checkpoint comparison and metric exports |
    | src/plot_evaluation.py | Plots from evaluation outputs and training logs |
    | src/predict_and_view.py | Saved-checkpoint predictions, JSON, browser viewer |
    | scripts/inference_hrp4k_rfdetr_small.py | Batch annotation output |
    | scripts/view_potholes.py | Dataset and annotation viewer |
    | scripts/plot_25_epoch_run.py | Run-specific graphs and cached image ROC |
    | data/finetune_hrp4k/ | Prepared train/valid images and COCO labels |
    | out/.../<run>/evaluation/<timestamp>/ | metrics.csv, results.json, curves.csv/json, splits.json, plots/ |
    | scripts/graphs/ | Saved figures and supporting numerical exports |
    | docs/pothole_detection_approaches_report.pdf | Earlier methodology proposal |

    The earlier proposal discusses YOLO, D-FINE, SAM 3, LoRA, tiling, and training from scratch. These are research alternatives, not implemented project features. The README references an original Colab notebook, but that notebook is absent from the inspected checkout.

    ## 12. Limitations and next steps
    The 41-image local validation set is small and has no negative images. Random image splitting does not guarantee separation by road, recording session, or near-duplicate scene. Local validation influences checkpoint selection, and HRP4K validation previously influenced selection of the public base checkpoint. These comparisons are validation results, not an untouched final test.

    For the next experiment, audit labels, include representative local negative images, group splits by road/session, and reserve an independent test set. Choose confidence on validation data according to acceptable missed-pothole and false-alarm rates, then freeze it before final testing. Compare encoder freezing and learning-rate changes as separate controlled experiments. No speed benchmark, independent-test score, or production-readiness claim is established by the saved artifacts.

    ## 13. Troubleshooting and reproducibility
    Missing local images: check original_coco.json paths relative to data/. Missing prepared splits: run training preparation with the intended dataset before evaluation. Missing checkpoint: correct MODEL_DIR, WEIGHTS, or --weights. Port in use: select --port 8001. GPU memory pressure: reduce BATCH_SIZE and adjust accumulation deliberately. Downloads: confirm connectivity and reuse cached metadata/images. Slow startup: model initialization and initial CUDA work occur before the training progress bar appears.

    Archive training_config.json, metrics.csv, checkpoints, prepared annotations, evaluation settings, and splits.json together. Record code revision and dependency lockfile. Existing scripts do not pin remote Hugging Face revisions for future downloads. data/ and out/ are gitignored, so a code checkout alone does not include training data or models.

    Documentation evidence: current source files, pyproject.toml, uv.lock, local COCO annotations, the saved run configuration, evaluation results, and scripts/graphs/summary.json. Metrics were read from existing artifacts; training and inference were not rerun to prepare this document.
