"""Local checks for dataset preservation and the two official-runtime adapters."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from detr_training import MODEL_CONFIGS, build_config, launch_training, prediction_command, prepare_annotations
from detr_worker import checkpoint_state, matching_state, prediction_rows, replace_scheduler_config


class DatasetTests(unittest.TestCase):
    def make_dataset(self, root):
        source = root / "source"
        for split in ("train", "valid"):
            folder = source / split
            folder.mkdir(parents=True)
            # Repeated source content retains distinct COCO image entries, just as in the shared dataset.
            for name in ("00001_repeat.jpg", "00002_repeat.jpg", "00003_empty.jpg"):
                (folder / name).write_bytes(b"image")
            coco = {"images": [{"id": i, "file_name": name, "width": 100, "height": 80}
                               for i, name in enumerate(("00001_repeat.jpg", "00002_repeat.jpg", "00003_empty.jpg"), 1)],
                    "categories": [{"id": 1, "name": "pothole"}],
                    "annotations": [{"id": i, "image_id": i, "category_id": 1,
                                     "bbox": [10, 10, 20, 30], "area": 600, "iscrowd": 0} for i in (1, 2)]}
            (folder / "_annotations.coco.json").write_text(json.dumps(coco))
        return source

    def test_conversion_preserves_split_entries_boxes_empty_images_and_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_dataset(root)
            before = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
            manifest = prepare_annotations(source, root / "converted")
            for split in ("train", "valid"):
                original = json.loads((source / split / "_annotations.coco.json").read_text())
                converted = json.loads((root / "converted" / f"{split}.json").read_text())
                self.assertEqual(converted["images"], original["images"])
                self.assertEqual([row["bbox"] for row in converted["annotations"]],
                                 [row["bbox"] for row in original["annotations"]])
                self.assertEqual({row["category_id"] for row in converted["annotations"]}, {0})
                self.assertEqual(manifest[split]["images"], 3)
                self.assertEqual(manifest[split]["boxes"], 2)
            self.assertEqual(before, {path: path.read_bytes() for path in before})
            (source / "valid/00003_empty.jpg").unlink()
            with self.assertRaises(FileNotFoundError):
                prepare_annotations(source, root / "invalid")
            self.assertFalse((root / "invalid").exists())
            with self.assertRaises(ValueError):
                prepare_annotations(source, source / "generated")

    def test_configuration_uses_exact_source_paths_and_model_specific_preprocessing(self):
        settings = SimpleNamespace(EPOCHS=100, RESOLUTION=896, REPO_DIR=Path("/official"),
                                   SOURCE_DATASET=Path("/dataset"), BATCH_SIZE=32, WORKERS=2,
                                   LEARNING_RATE=2e-4)
        for kind, epochs_key in (("dfine", "epochs"), ("deimv2", "epoches")):
            cfg = build_config(kind, settings, Path("/run"))
            self.assertEqual(cfg[epochs_key], 100)
            self.assertEqual(cfg["num_classes"], 1)
            self.assertFalse(cfg["remap_mscoco_category"])
            self.assertEqual(cfg["checkpoint_freq"], 1)
            self.assertEqual(cfg["optimizer"]["params"], [])
            for split, key in (("train", "train_dataloader"), ("valid", "val_dataloader")):
                loader = cfg[key]
                self.assertEqual(loader["dataset"]["img_folder"], f"/dataset/{split}")
                self.assertFalse(loader["drop_last"])
                ops = loader["dataset"]["transforms"]["ops"]
                self.assertEqual(any(op["type"] == "Normalize" for op in ops), kind == "deimv2")
                self.assertIn({"type": "Resize", "size": [896, 896]}, ops)
                if split == "valid":
                    self.assertFalse(any(op["type"].startswith("Random") for op in ops))
            self.assertIsNone(cfg["train_dataloader"]["collate_fn"]["base_size_repeat"])
            self.assertGreater(cfg["train_dataloader"]["collate_fn"]["stop_epoch"], 100)
            self.assertEqual(cfg["train_dataloader"]["dataset"]["transforms"]["policy"]["name"], "default")


class RuntimeTests(unittest.TestCase):
    def test_missing_source_is_distinguished_from_missing_python(self):
        for kind in MODEL_CONFIGS:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                repo = root / "vendor"
                python = repo / ".venv/bin/python"
                python.parent.mkdir(parents=True)
                python.touch()
                settings = SimpleNamespace(
                    EPOCHS=100, RESOLUTION=896, BATCH_SIZE=32, WORKERS=2,
                    LEARNING_RATE=2e-4, FREEZE_BACKBONE=True, SEED=42,
                    REPO_DIR=repo, PYTHON=python, OUTPUT_DIR=root / "runs",
                    SOURCE_DATASET=DatasetTests().make_dataset(root),
                )
                with patch.object(sys, "argv", ["finetune.py"]), patch("detr_training.subprocess.run") as run:
                    with self.assertRaises(SystemExit) as error:
                        launch_training(kind, settings)
                    self.assertIn(str(repo / MODEL_CONFIGS[kind]), str(error.exception))
                    self.assertIn("only installs dependencies", str(error.exception))
                    run.assert_not_called()

                model_config = repo / MODEL_CONFIGS[kind]
                model_config.parent.mkdir(parents=True)
                model_config.touch()
                python.unlink()
                with patch.object(sys, "argv", ["finetune.py"]), patch("detr_training.subprocess.run") as run:
                    with self.assertRaises(SystemExit) as error:
                        launch_training(kind, settings)
                    self.assertIn(f"Python executable is missing: {python}", str(error.exception))
                    run.assert_not_called()

                python.touch()
                with patch.object(sys, "argv", ["finetune.py"]), patch("detr_training.subprocess.run") as run:
                    launch_training(kind, settings)
                    self.assertEqual(run.call_args.args[0][0], str(python))
                    self.assertEqual(run.call_args.kwargs["cwd"], repo)


class CheckpointTests(unittest.TestCase):
    def test_scheduler_steps_after_json_yaml_loading_and_resume(self):
        import torch
        import yaml

        for eta_min in (5e-6, 2e-6):
            with self.subTest(eta_min=eta_min):
                # Preparation writes JSON, which the worker reads using PyYAML.
                requested = yaml.safe_load(json.dumps({"lr_scheduler": {
                    "type": "CosineAnnealingLR", "T_max": 100, "eta_min": eta_min,
                }}))
                for resumed in (False, True):
                    if resumed:
                        requested = yaml.safe_load(yaml.safe_dump(cfg.yaml_cfg))
                    cfg = SimpleNamespace(yaml_cfg={})
                    replace_scheduler_config(cfg, requested)
                    arguments = dict(cfg.yaml_cfg["lr_scheduler"])
                    arguments.pop("type")
                    parameter = torch.nn.Parameter(torch.zeros(1))
                    optimizer = torch.optim.AdamW([parameter], lr=5e-4)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **arguments)
                    for _ in range(2):
                        parameter.sum().backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                    self.assertIsInstance(scheduler.eta_min, float)
                    self.assertEqual(scheduler.eta_min, eta_min)
                    self.assertLess(optimizer.param_groups[0]["lr"], 5e-4)

    def test_changed_scheduler_type_does_not_keep_inherited_arguments(self):
        import torch

        requested = {"lr_scheduler": {"type": "CosineAnnealingLR", "T_max": 100, "eta_min": 2e-6}}
        cfg = SimpleNamespace(yaml_cfg={"lr_scheduler": dict(requested["lr_scheduler"], milestones=[500], gamma=0.1)})
        replace_scheduler_config(cfg, requested)
        arguments = dict(cfg.yaml_cfg["lr_scheduler"])
        arguments.pop("type")
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **arguments)
        self.assertEqual(scheduler.T_max, 100)
        self.assertNotIn("milestones", cfg.yaml_cfg["lr_scheduler"])

    def test_class_head_shape_changes_are_allowed_but_missing_backbones_are_rejected(self):
        feature = SimpleNamespace(shape=(3, 3))
        small_head, coco_head = SimpleNamespace(shape=(1, 3)), SimpleNamespace(shape=(80, 3))
        model = {"backbone.feature": feature, "decoder.head": small_head}
        matched = matching_state(model, {"backbone.feature": feature, "decoder.head": coco_head})
        self.assertEqual(matched, {"backbone.feature": feature})
        with self.assertRaises(ValueError):
            matching_state(model, {"decoder.head": coco_head})
        self.assertEqual(checkpoint_state({"ema": {"module": "ema"}, "model": "plain"}), "ema")
        self.assertEqual(checkpoint_state({"model": "plain"}), "plain")

    def test_prediction_filter_retains_original_coordinates_and_empty_results(self):
        rows = prediction_rows([0, 0, 1], [[10, 20, 100, 200], [1, 2, 3, 4], [4, 5, 6, 7]],
                               [0.9, 0.2, 0.99], 0.3)
        self.assertEqual(rows, [{"label": "pothole", "score": 0.9,
                                 "box_xyxy": [10.0, 20.0, 100.0, 200.0]}])
        self.assertEqual(prediction_rows([], [], [], 0.3), [])

    def test_predictions_use_saved_runtime_and_reject_wrong_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "run.json").write_text(json.dumps({"kind": "dfine", "python": "/separate/env/python"}))
            command = prediction_command("dfine", run / "last.pth", run / "in.json", run / "out.json", "cpu")
            self.assertEqual(command[0], "/separate/env/python")
            self.assertIn(str(run / "last.pth"), command)
            with self.assertRaises(ValueError):
                prediction_command("deimv2", run / "last.pth", run / "in.json", run / "out.json", "cpu")


if __name__ == "__main__":
    unittest.main()
