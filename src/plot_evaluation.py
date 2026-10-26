"""Plot saved evaluation results alongside the model's existing training logs."""

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SPLITS = ("train_my_dataset", "val_my_dataset", "train_hrp", "val_hrp")
METRICS = ("F1", "precision", "recall", "mAP50", "mAP75", "mAP50_95", "mAR100")


def save_figure(figure, path):
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def checkpoint_comparison(rows, output_dir):
    models = list(dict.fromkeys(row["model"] for row in rows))
    lookup = {(row["model"], row["split"]): row for row in rows}
    x = range(len(models))
    figure, axes = plt.subplots(3, 3, figsize=(18, 15), layout="constrained")
    for axis, metric in zip(axes.flat, METRICS):
        for split in SPLITS:
            values = [lookup.get((model, split), {}).get(metric) for model in models]
            axis.plot(x, [value if value is not None else float("nan") for value in values],
                      marker="o", markersize=3, label=split)
        axis.set(title=metric, ylabel="Percent", ylim=(0, 100))
        axis.set_xticks(list(x), models, rotation=70, ha="right", fontsize=7)
        axis.grid(alpha=0.2)
    for axis in list(axes.flat)[len(METRICS):]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    figure.suptitle("Base model and every saved checkpoint · identical evaluation settings")
    save_figure(figure, output_dir / "metrics_comparison.png")

    figure, axes = plt.subplots(2, 2, figsize=(16, 10), layout="constrained")
    for axis, split in zip(axes.flat, SPLITS):
        for metric in ("TP", "FP", "FN"):
            axis.plot(x, [lookup.get((model, split), {}).get(metric, float("nan"))
                          for model in models], marker="o", markersize=3, label=metric)
        axis.set(title=split, ylabel="Number of detections / missed targets")
        axis.set_xticks(list(x), models, rotation=70, ha="right", fontsize=7)
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("Detection counts at the configured confidence and IoU thresholds")
    save_figure(figure, output_dir / "detection_counts.png")


def confidence_plots(curves, settings, output_dir):
    models = list(curves)
    colors = {model: "black" if model == "base" else plt.get_cmap("turbo")(
        index / max(len(models) - 1, 1)) for index, model in enumerate(models)}
    iou = settings["iou_for_precision_recall_f1"]
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    legend = {}
    for axis, split in zip(axes.flat, SPLITS):
        for model, model_curves in curves.items():
            curve = model_curves.get(split)
            if curve is None:
                continue
            # No-prediction points have undefined precision; omit them from PR.
            valid = [index for index, count in enumerate(curve["detections"]) if count]
            recall = [curve["recall"][index] for index in reversed(valid)]
            precision = [curve["precision"][index] for index in reversed(valid)]
            line, = axis.plot(recall, precision, color=colors[model],
                              linestyle="--" if model == "base" else "-", label=model)
            legend[model] = line
        axis.set(title=split, xlabel="Recall (%)", ylabel="Precision (%)",
                 xlim=(0, 100), ylim=(0, 100))
        axis.grid(alpha=0.2)
    figure.suptitle(f"Precision–recall curves · IoU ≥ {iou:.2f}")
    figure.legend(legend.values(), legend.keys(), loc="outside lower center",
                  ncol=min(4, len(models)), fontsize=8)
    save_figure(figure, output_dir / "precision_recall.png")

    figure, axes = plt.subplots(4, 3, figsize=(16, 15), layout="constrained")
    for row, split in enumerate(SPLITS):
        for column, metric in enumerate(("precision", "recall", "F1")):
            axis = axes[row, column]
            for model, model_curves in curves.items():
                curve = model_curves.get(split)
                if curve is not None:
                    axis.plot(curve["confidence"], curve[metric], color=colors[model],
                              linestyle="--" if model == "base" else "-")
            axis.axvline(settings["confidence_for_precision_recall_f1"],
                         color="gray", linestyle=":", linewidth=1)
            axis.set(title=f"{split} · {metric}", xlabel="Confidence threshold",
                     ylabel="Percent", xlim=(0, 1), ylim=(0, 100))
            axis.grid(alpha=0.2)
    figure.suptitle(f"Confidence sweeps · IoU ≥ {iou:.2f} · dotted line = reported threshold")
    figure.legend(legend.values(), legend.keys(), loc="outside lower center",
                  ncol=min(4, len(models)), fontsize=8)
    save_figure(figure, output_dir / "confidence_curves.png")


def training_history(model_dir, output_dir):
    paths = list(model_dir.glob("metrics.csv"))
    paths += sorted(model_dir.glob("lightning_logs/version_*/metrics.csv"))
    if not paths:
        print("No existing training metrics.csv found; skipping the training-history graph.")
        return
    groups = [
        ("Training losses", ["train/loss", "train/loss_bbox", "train/loss_ce", "train/loss_giou"], 1),
        ("Logged validation P/R/F1", ["val/precision", "val/recall", "val/F1"], 100),
        ("Logged validation mAP", ["val/mAP_50", "val/mAP_50_95", "val/mAP_75"], 100),
        ("Logged validation mean recall", ["val/mAR", "val/ema_mAR"], 100),
        ("Learning rates", ["train/lr", "train/lr_min", "train/lr_max"], 1),
        ("Training classification error", ["train/class_error"], 1),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(17, 10), layout="constrained")
    for path in paths:
        with path.open() as file:
            records = list(csv.DictReader(file))
        for axis, (title, keys, scale) in zip(axes.flat, groups):
            for key in keys:
                # Lightning writes sparse rows; keep the last logged value per epoch.
                values = {}
                for record in records:
                    if record.get(key) and record.get("epoch"):
                        value = float(record[key])
                        if math.isfinite(value):
                            values[int(float(record["epoch"])) + 1] = value * scale
                if values:
                    epochs = sorted(values)
                    label = key if len(paths) == 1 else f"{path.parent.name}: {key}"
                    axis.plot(epochs, [values[epoch] for epoch in epochs],
                              marker="o", markersize=3, label=label)
            axis.set(title=title, xlabel="Epoch", ylabel="Percent" if scale == 100 else "Value")
            axis.grid(alpha=0.2)
    for axis in axes.flat:
        if axis.lines:
            axis.legend(fontsize=8)
        else:
            axis.text(0.5, 0.5, "No logged values", ha="center", transform=axis.transAxes)
    figure.suptitle("Existing training history · RF-DETR validation thresholds are as originally logged")
    save_figure(figure, output_dir / "training_history.png")


def create_plots(evaluation_dir, model_dir):
    evaluation_dir, model_dir = Path(evaluation_dir), Path(model_dir)
    saved = json.loads((evaluation_dir / "results.json").read_text())
    output_dir = evaluation_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_comparison(saved["results"], output_dir)
    curves_path = evaluation_dir / "curves.json"
    if curves_path.is_file():
        confidence_plots(json.loads(curves_path.read_text()), saved["settings"], output_dir)
    else:
        print("No saved confidence sweeps; PR and confidence plots require a new evaluation.")
    training_history(model_dir, output_dir)
    print(f"Graphs saved to: {output_dir}")
