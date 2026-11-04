"""CPU-only checks; run this file directly without installing ML packages."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/yolo"))

import config
import finetune_yolo26_small as training
from prepare_dataset import prepare_dataset, yolo_box


class DatasetTests(unittest.TestCase):
    def test_box_coordinates_and_roundoff(self):
        self.assertEqual(yolo_box([10, 20, 40, 60], 100, 200), (0.3, 0.25, 0.4, 0.3))
        box = yolo_box([-2.2737367544323206e-13, 20, 40, 60], 100, 200)
        self.assertAlmostEqual(box[0], 0.2)
        for invalid in ([-1, 0, 10, 10], [0, 0, 0, 10], [90, 0, 20, 10],
                        [float("nan"), 0, 10, 10]):
            with self.assertRaises(ValueError):
                yolo_box(invalid, 100, 200)

    def test_conversion_preserves_entries_and_empty_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination = root / "coco", root / "yolo"
            for split in ("train", "valid"):
                folder = source / split
                folder.mkdir(parents=True)
                images = []
                for image_id in (1, 2, 3):
                    name = f"{image_id}_repeated.jpg"
                    (folder / name).write_bytes(split.encode())
                    images.append({"id": image_id, "file_name": name, "width": 100, "height": 200})
                (folder / "_annotations.coco.json").write_text(json.dumps({
                    "images": images,
                    "categories": [{"id": 7, "name": "pothole"}],
                    "annotations": [{"image_id": 1, "category_id": 7, "bbox": [10, 20, 40, 60]}],
                }))
            prepare_dataset(source, destination)
            for split in ("train", "val"):
                entries = (destination / f"{split}.txt").read_text().splitlines()
                self.assertEqual(len(entries), 3)
                self.assertEqual(len(set(entries)), 3)
                self.assertTrue(all(Path(entry).is_symlink() for entry in entries))
                source_split = "valid" if split == "val" else split
                self.assertTrue(all(Path(entry).resolve().parent == source / source_split
                                    for entry in entries))
                self.assertEqual(len({Path(entry).read_bytes() for entry in entries}), 1)
                self.assertEqual((destination / "labels" / split / "00001.txt").read_text(),
                                 "0 0.3000000000 0.2500000000 0.4000000000 0.3000000000\n")
                self.assertEqual((destination / "labels" / split / "00002.txt").read_text(), "")
            # Rebuilding must not include stale entries or retain label caches.
            (destination / "images/train/stale.jpg").write_bytes(b"stale")
            cache = destination / "labels/train.cache"
            cache.write_text("old cache")
            prepare_dataset(source, destination)
            self.assertNotIn("stale", (destination / "train.txt").read_text())
            self.assertFalse(cache.exists())
            (source / "train/1_repeated.jpg").unlink()
            with self.assertRaises(FileNotFoundError):
                prepare_dataset(source, destination)


class TrainingTests(unittest.TestCase):
    def test_training_settings_without_loading_models(self):
        for cuda, device in ((False, "cpu"), (True, 0)):
            with self.subTest(cuda=cuda), tempfile.TemporaryDirectory() as temporary:
                yolo, settings = MagicMock(), MagicMock()
                modules = {
                    "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda)),
                    "wandb": None,  # Training must work without this package.
                    "ultralytics": SimpleNamespace(YOLO=yolo, settings=settings),
                }
                with patch.dict(sys.modules, modules), patch.object(training, "OUTPUT_DIR", Path(temporary)), \
                        patch.object(training, "prepare_dataset", return_value=Path("dataset.yaml")):
                    training.main()
                arguments = yolo.return_value.train.call_args.kwargs
                self.assertEqual(arguments["device"], device)
                self.assertEqual(arguments["data"], "dataset.yaml")
                self.assertEqual(arguments["optimizer"], "AdamW")
                self.assertEqual(arguments["lr0"], 1e-3)
                self.assertEqual(arguments["batch"], arguments["nbs"])
                self.assertEqual(arguments["freeze"], 11)
                self.assertEqual(arguments["patience"], 10)
                self.assertEqual(arguments["save_period"], 1)
                self.assertEqual(arguments["imgsz"], 896)
                yolo.assert_called_once_with("yolo26s.pt")
                settings.update.assert_called_once_with({"wandb": False})


if __name__ == "__main__":
    unittest.main()
