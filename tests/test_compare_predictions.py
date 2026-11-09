"""Check inference adapters and saved viewer output without loading ML checkpoints."""

import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("compare_predictions", ROOT / "src/compare_predictions.py")
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


def tensor(values):
    return SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: values))


class ComparisonTests(unittest.TestCase):
    def test_default_launch_selects_all_four_models_and_explicit_selection_is_respected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs"
            inputs.mkdir()
            Image.new("RGB", (20, 20)).save(inputs / "image.jpg")
            weights = {}
            for kind in viewer.MODEL_NAMES:
                run = root / kind
                run.mkdir()
                weights[kind] = run / "best.pth"
                weights[kind].touch()
                (run / "run.json").write_text(json.dumps({"kind": kind, "python": sys.executable}))
                (run / "resolved_config.yaml").touch()
            defaults = {f"{kind.upper()}_WEIGHTS": path for kind, path in weights.items()}
            for selected in ([], ["--models", "yolo"]):
                with self.subTest(selected=selected), patch.multiple(viewer, **defaults), \
                        patch.object(sys, "argv", ["compare_predictions.py", "--input-dir", str(inputs),
                                                  "--output-dir", str(root / "results"), *selected]), \
                        patch.object(viewer, "run_inference", return_value=root / "results") as infer, \
                        patch.object(viewer, "serve_viewer") as serve:
                    viewer.main()
                    args, paths = infer.call_args.args
                    self.assertEqual(args.models, ["yolo"] if selected else list(viewer.MODEL_NAMES))
                    self.assertEqual(paths, [inputs / "image.jpg"])
                    for kind in args.models:
                        self.assertEqual(getattr(args, f"{kind}_weights"), weights[kind])
                    serve.assert_called_once_with(root / "results", viewer.PORT)

    def test_split_membership_uses_prepared_data_and_deduplicates_repeats(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, local = root / "prepared", root / "local"
            for split, names in (("train", ["00001_seen.jpg", "00002_seen.jpg"]),
                                 ("valid", ["00001_held_out.jpg"])):
                folder = dataset / split
                folder.mkdir(parents=True)
                (folder / "_annotations.coco.json").write_text(json.dumps({
                    "images": [{"file_name": name} for name in names],
                }))
            lookup = viewer.load_split_lookup(dataset, root / "missing.json", local)
            self.assertEqual(lookup[(local / "seen.jpg").resolve()], "train")
            self.assertEqual(lookup[(local / "held_out.jpg").resolve()], "valid")
            self.assertEqual(lookup[(dataset / "valid/00001_held_out.jpg").resolve()], "valid")
            self.assertNotIn((root / "unrelated/seen.jpg").resolve(), lookup)

    def test_refresh_saved_viewer_adds_splits_without_loading_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = {"input_dir": str(root), "images": [
                {"name": "train.jpg"}, {"name": "valid.jpg"}, {"name": "other.jpg"},
                {"name": "recorded.jpg", "split": "valid"},
            ]}
            lookup = {root / "train.jpg": "train", root / "valid.jpg": "valid",
                      root / "recorded.jpg": "train"}
            (root / "predictions.json").write_text(json.dumps(results))
            with patch.object(viewer, "load_split_lookup", return_value=lookup), \
                    patch.object(viewer, "load_model") as load:
                viewer.write_results(root)
            load.assert_not_called()
            saved = json.loads((root / "predictions.json").read_text())
            self.assertEqual([item["split"] for item in saved["images"]],
                             ["train", "valid", "unknown", "valid"])
            self.assertIn('data-split="train"', (root / "index.html").read_text())
            self.assertIn('data-split="valid"', (root / "index.html").read_text())

    def test_adapters_preserve_coordinates_and_filter_class_names(self):
        model = MagicMock()
        image = Image.new("RGB", (200, 100))
        model.predict.return_value = SimpleNamespace(
            xyxy=[[11, 12, 30, 40], [1, 2, 5, 6], [50, 20, 100, 90]],
            confidence=[0.4, 0.99, 0.8],
            data={"class_name": ["pothole", "car", "Pothole"]},
        )
        rf = viewer.predict(model, "rfdetr", image, 0.3, "cpu")
        self.assertEqual([item["score"] for item in rf], [0.8, 0.4])
        self.assertEqual(rf[0]["box_xyxy"], [50, 20, 100, 90])
        model.predict.assert_called_once_with(image, threshold=0.3)

        model.reset_mock()
        model.predict.return_value = [SimpleNamespace(
            boxes=SimpleNamespace(xyxy=tensor([[11, 12, 30, 40], [1, 2, 5, 6]]),
                                  conf=tensor([0.4, 0.99]), cls=tensor([0, 1])),
            names={0: "pothole", 1: "car"},
        )]
        yolo = viewer.predict(model, "yolo", image, 0.3, "cpu")
        self.assertEqual(yolo, [rf[1]])
        model.predict.assert_called_once_with(source=image, conf=0.3, imgsz=896,
                                             rect=False, device="cpu", verbose=False)

        model.predict.return_value[0].boxes = SimpleNamespace(
            xyxy=tensor([]), conf=tensor([]), cls=tensor([]))
        self.assertEqual(viewer.predict(model, "yolo", image, 0.3, "cpu"), [])

    def test_pipeline_pairs_nested_images_and_keeps_empty_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs"
            paths = [inputs / "a/same.jpg", inputs / "b/same.jpg"]
            for path, color in zip(paths, ("white", "black")):
                path.parent.mkdir(parents=True)
                Image.new("RGB", (200, 100), color).save(path)
            args = SimpleNamespace(models=["rfdetr", "yolo"], device="cpu", output_dir=root / "results", input_dir=inputs,
                                   confidence=0.3, rfdetr_weights=root / "rf.ckpt",
                                   yolo_weights=root / "yolo.pt")
            detection = {"box_xyxy": [20, 20, 100, 80], "score": 0.75, "label": "pothole"}
            fake_torch = SimpleNamespace(inference_mode=nullcontext)
            modules = {"torch": fake_torch, "rfdetr": SimpleNamespace(), "ultralytics": SimpleNamespace()}
            with patch.dict(sys.modules, modules), patch.object(viewer, "load_model") as load, \
                    patch.object(viewer, "predict", side_effect=[[detection], [], [], [detection]]):
                output = viewer.run_inference(args, paths)
            self.assertEqual([call.args[0] for call in load.call_args_list], ["rfdetr", "yolo"])
            results = json.loads((output / "predictions.json").read_text())
            self.assertEqual([item["name"] for item in results["images"]], ["a/same.jpg", "b/same.jpg"])
            self.assertEqual(results["images"][0]["yolo"]["detections"], [])
            self.assertEqual(results["images"][1]["rfdetr"]["detections"], [])
            for item in results["images"]:
                for relative in (item["original"], item["rfdetr"]["image"], item["yolo"]["image"]):
                    with Image.open(output / relative) as saved:
                        self.assertEqual(saved.size, (200, 100))
            original = viewer.read_image(output / results["images"][0]["original"])
            annotated = viewer.read_image(output / results["images"][0]["rfdetr"]["image"])
            self.assertNotEqual(original.getpixel((20, 60)), annotated.getpixel((20, 60)))
            page = (output / "index.html").read_text()
            self.assertNotIn("__RESULTS__", page)
            embedded = page.split('<script id="results" type="application/json">')[1].split('</script>')[0]
            self.assertEqual(json.loads(embedded), results)

    def test_exif_orientation_is_applied_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rotated.jpg"
            image = Image.new("RGB", (100, 200))
            exif = Image.Exif()
            exif[274] = 6  # Rotate 90 degrees clockwise for display.
            image.save(path, exif=exif)
            oriented = viewer.read_image(path)
            self.assertEqual(oriented.size, (200, 100))
            self.assertEqual(oriented.mode, "RGB")

    def test_new_models_predict_in_separate_runtimes_and_pair_by_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs"
            inputs.mkdir()
            paths = [inputs / "image.jpg"]
            Image.new("RGB", (200, 100), "white").save(paths[0])
            args = SimpleNamespace(models=["dfine", "deimv2"], device="cpu", output_dir=root / "results",
                                   input_dir=inputs, confidence=0.3,
                                   dfine_weights=root / "dfine/last.pth", deimv2_weights=root / "deim/last.pth")
            calls = []

            def run_worker(command, check):
                calls.append(command)
                output = Path(command[command.index("--output") + 1])
                output.write_text(json.dumps([[]]))

            def command(kind, checkpoint, request, output, device):
                return [kind, "--output", str(output)]

            with patch.dict(sys.modules, {"torch": SimpleNamespace()}), \
                    patch("detr_training.prediction_command", side_effect=command), \
                    patch.object(viewer.subprocess, "run", side_effect=run_worker), \
                    patch.object(viewer, "load_model") as load:
                output = viewer.run_inference(args, paths)
            load.assert_not_called()
            self.assertEqual([call[0] for call in calls], ["dfine", "deimv2"])
            results = json.loads((output / "predictions.json").read_text())
            self.assertEqual(set(results["weights"]), {"dfine", "deimv2"})
            for kind in args.models:
                item = results["images"][0][kind]
                self.assertEqual(item["detections"], [])
                self.assertTrue((output / item["image"]).is_file())


if __name__ == "__main__":
    unittest.main()
