"""Run saved RF-DETR and YOLO checkpoints and browse their predictions side by side."""

import argparse
import gc
import json
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm

from pothole_data import DATA_DIR, split_local_samples


ROOT = Path(__file__).resolve().parents[1]
# Edit these paths to compare other saved checkpoints.
RFDETR_WEIGHTS = ROOT / "out/hrp4k-rfdetr-small-finetuned/new/checkpoint_19.ckpt"
YOLO_WEIGHTS = ROOT / "out/yolo/1/best.pt"
INPUT_DIR = ROOT / "data/processed_images"
CONFIDENCE = 0.30
RESOLUTION = 896
PORT = 8000
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MODEL_NAMES = {"rfdetr": "RF-DETR", "yolo": "YOLO"}
COLORS = {"rfdetr": "#c53d19", "yolo": "#1262c4"}


PAGE = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RF-DETR / YOLO predictions</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0 auto; padding: 24px; max-width: 1800px; background: #f4f5f7;
         color: #192434; font: 16px system-ui, sans-serif; }
  h1 { margin: 0 0 8px; } p { color: #526176; }
  .controls { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin: 18px 0; }
  input, select, button { font: inherit; padding: 10px; border: 1px solid #bbc4cf;
                         border-radius: 6px; background: white; }
  input[type=search] { flex: 1; min-width: 180px; }
  #splits { display: flex; gap: 16px; margin: 20px 0; flex-wrap: wrap; }
  #splits button { flex: 1; min-width: 180px; padding: 18px; text-align: left; cursor: pointer; }
  #splits button[aria-pressed=true] { border: 2px solid #1262c4; background: #eaf2ff; }
  #splits strong { display: block; font-size: 20px; margin-bottom: 6px; }
  #panels { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; }
  #panels[hidden] { display: none; }
  article { min-width: 0; padding: 16px; background: white; border-radius: 10px; }
  h2 { margin-top: 0; } #rfdetr h2 { color: #c53d19; } #yolo h2 { color: #1262c4; }
  article img { width: 100%; display: block; background: #101820; }
  article a { display: block; } small { display: block; overflow-wrap: anywhere; color: #526176; }
  ol { max-height: 220px; overflow: auto; padding-left: 28px; }
  li { padding: 3px 0; } #filename { overflow-wrap: anywhere; }
  @media (max-width: 800px) { #panels { grid-template-columns: 1fr; } body { padding: 12px; } }
</style>
<h1>RF-DETR / YOLO predictions</h1>
<p id="summary"></p>
<div id="splits" role="group" aria-label="Dataset split">
  <button data-split="train" aria-pressed="false"><strong>Training data</strong><span></span></button>
  <button data-split="valid" aria-pressed="false"><strong>Validation data</strong><span></span></button>
</div>
<div class="controls">
  <input id="search" type="search" placeholder="Search filenames" aria-label="Search filenames">
  <select id="filter" aria-label="Filter predictions">
    <option value="all">All images</option>
    <option value="positive">Either model detects potholes</option>
    <option value="negative">Neither model detects potholes</option>
    <option value="different">Different detection counts</option>
  </select>
  <label><input id="original" type="checkbox"> Show original</label>
</div>
<div class="controls">
  <button id="previous">← Previous</button>
  <select id="image-picker" aria-label="Choose image"></select>
  <button id="next">Next →</button>
  <span id="position" role="status"></span>
</div>
<h3 id="filename"></h3>
<main id="panels">
  <article id="rfdetr"><h2>RF-DETR</h2><small></small><p></p>
    <a target="_blank" rel="noopener"><img alt="RF-DETR predictions"></a><ol></ol></article>
  <article id="yolo"><h2>YOLO</h2><small></small><p></p>
    <a target="_blank" rel="noopener"><img alt="YOLO predictions"></a><ol></ol></article>
</main>
<p>Use ← / → to browse. Click an image to open it at full size.
Box numbers match the predictions listed below each image.</p>
<script id="results" type="application/json">__RESULTS__</script>
<script>
  const results = JSON.parse(document.querySelector('#results').textContent);
  const splitImages = results.images.filter(item => ['train', 'valid'].includes(item.split));
  const search = document.querySelector('#search'), filter = document.querySelector('#filter');
  const original = document.querySelector('#original'), picker = document.querySelector('#image-picker');
  const previous = document.querySelector('#previous'), next = document.querySelector('#next');
  const splitButtons = [...document.querySelectorAll('#splits button')];
  const splitLabels = {train: 'Training', valid: 'Validation'};
  let selectedSplit = ['train', 'valid'].find(split => splitImages.some(item => item.split === split)) || 'train';
  let shown = [], current = 0;
  document.querySelector('#summary').textContent =
    `${splitImages.length} images · confidence ≥ ${results.confidence} · input size ${results.resolution}`;
  function render() {
    const item = shown[current];
    picker.value = String(current);
    previous.disabled = !item || current === 0;
    next.disabled = !item || current === shown.length - 1;
    document.querySelector('#position').textContent = item ? `${current + 1} / ${shown.length}` : 'No matching images';
    document.querySelector('#filename').textContent = item ? `${splitLabels[item.split]} · ${item.name}` : '';
    document.querySelector('#panels').hidden = !item;
    if (!item) return;
    for (const model of ['rfdetr', 'yolo']) {
      const panel = document.querySelector(`#${model}`), prediction = item[model];
      panel.querySelector('small').textContent = results.weights[model];
      panel.querySelector('p').textContent = `${prediction.detections.length} pothole detections`;
      const url = original.checked ? item.original : prediction.image;
      panel.querySelector('img').src = url;
      panel.querySelector('a').href = url;
      panel.querySelector('ol').replaceChildren(...prediction.detections.map(detection => {
        const li = document.createElement('li');
        li.textContent = `${detection.label}: ${(detection.score * 100).toFixed(1)}% · box [${detection.box_xyxy.map(Math.round).join(', ')}]`;
        return li;
      }));
    }
  }
  function applyFilters() {
    const term = search.value.toLowerCase();
    shown = splitImages.filter(item => {
      const a = item.rfdetr.detections.length, b = item.yolo.detections.length;
      return item.split === selectedSplit && item.name.toLowerCase().includes(term) &&
        (filter.value === 'all' || (filter.value === 'positive' && a + b > 0) ||
         (filter.value === 'negative' && a + b === 0) || (filter.value === 'different' && a !== b));
    });
    for (const button of splitButtons) {
      button.setAttribute('aria-pressed', String(button.dataset.split === selectedSplit));
    }
    current = 0;
    picker.replaceChildren(...shown.map((item, index) => new Option(item.name, String(index))));
    render();
  }
  search.addEventListener('input', applyFilters); filter.addEventListener('change', applyFilters);
  for (const button of splitButtons) {
    const count = splitImages.filter(item => item.split === button.dataset.split).length;
    button.querySelector('span').textContent = `${count} images`;
    button.addEventListener('click', () => { selectedSplit = button.dataset.split; applyFilters(); });
  }
  original.addEventListener('change', render);
  picker.addEventListener('change', () => { current = Number(picker.value); render(); });
  previous.addEventListener('click', () => { if (current > 0) { current--; render(); } });
  next.addEventListener('click', () => { if (current + 1 < shown.length) { current++; render(); } });
  document.addEventListener('keydown', event => {
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault(); (event.key === 'ArrowLeft' ? previous : next).click();
    }
  });
  applyFilters();
</script>
</html>
"""


def load_split_lookup(dataset_dir=DATA_DIR / "finetune_hrp4k",
                      annotations=DATA_DIR / "annotations/original_coco.json",
                      local_dir=DATA_DIR / "processed_images"):
    """Use prepared training membership, falling back to the shared seeded split."""
    lookup = {}
    split_files = {split: dataset_dir / split / "_annotations.coco.json" for split in ("train", "valid")}
    if all(path.is_file() for path in split_files.values()):
        for split, path in split_files.items():
            for item in json.loads(path.read_text())["images"]:
                prepared = path.parent / item["file_name"]
                candidates = [prepared.resolve()]
                # Preparation prefixes each source filename with its numeric image ID.
                prefix, separator, original_name = Path(item["file_name"]).name.partition("_")
                if separator and prefix.isdigit():
                    candidates.append((local_dir / original_name).resolve())
                for candidate in candidates:
                    previous = lookup.setdefault(candidate, split)
                    if previous != split:
                        lookup[candidate] = "unknown"
        return lookup
    if annotations.is_file():
        samples = [((annotations.parent.parent / item["file_name"]).resolve(), [])
                   for item in json.loads(annotations.read_text())["images"]]
        if len(samples) >= 2:
            train, valid = split_local_samples(samples)
            lookup.update((path, "train") for path, _ in train)
            lookup.update((path, "valid") for path, _ in valid)
    return lookup


def write_results(results_dir, results=None):
    """Refresh saved viewers without rerunning inference; retain recorded split labels."""
    if results is None:
        results = json.loads((results_dir / "predictions.json").read_text(encoding="utf-8"))
    missing = [item for item in results["images"] if "split" not in item]
    if missing:
        lookup = load_split_lookup()
        for item in missing:
            path = (Path(results["input_dir"]) / item["name"]).resolve()
            item["split"] = lookup.get(path, "unknown")
    (results_dir / "predictions.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    # Inline JSON lets the saved viewer also work directly from a file:// URL.
    embedded = json.dumps(results).replace("<", "\\u003c")
    (results_dir / "index.html").write_text(PAGE.replace("__RESULTS__", embedded), encoding="utf-8")


def read_image(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def annotate(image, detections, color):
    """Draw every retained prediction, numbered to match its list in the viewer."""
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(16, image.width // 65))
    for index, detection in enumerate(detections, 1):
        x1, y1, x2, y2 = detection["box_xyxy"]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(3, image.width // 500))
        label = f"{index}: {detection['label']} {detection['score']:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        label_x = max(0, min(x1, image.width - (right - left) - 8))
        label_y = max(0, y1 - (bottom - top) - 8)
        draw.rectangle((label_x, label_y, label_x + right - left + 8,
                        label_y + bottom - top + 8), fill=color)
        draw.text((label_x + 4 - left, label_y + 4 - top), label, fill="white", font=font)
    return image


def load_model(kind, weights, device):
    if kind == "rfdetr":
        from rfdetr import RFDETRSmall

        model = RFDETRSmall.from_checkpoint(str(weights), device=device, resolution=RESOLUTION)
        names = model.class_names
    else:
        from ultralytics import YOLO

        model = YOLO(str(weights))
        names = model.names.values() if isinstance(model.names, dict) else model.names
    if "pothole" not in [str(name).strip().lower() for name in names]:
        raise ValueError(f"{MODEL_NAMES[kind]} checkpoint has no pothole class: {names}")
    return model


def predict(model, kind, image, confidence, device):
    """Normalize the two model APIs to original-image xyxy boxes and confidence scores."""
    if kind == "rfdetr":
        result = model.predict(image, threshold=confidence)
        rows = zip(result.xyxy, result.confidence, result.data["class_name"])
    else:
        result = model.predict(source=image, conf=confidence, imgsz=RESOLUTION,
                               rect=False, device=device, verbose=False)[0]
        rows = zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                   [result.names[int(class_id)] for class_id in result.boxes.cls.cpu().tolist()])
    return sorted(
        [{"box_xyxy": [float(value) for value in box], "score": float(score), "label": "pothole"}
         for box, score, name in rows if str(name).strip().lower() == "pothole"],
        key=lambda item: item["score"], reverse=True,
    )


def run_inference(args, paths):
    # Validate both dependencies before starting a potentially long inference run.
    try:
        import torch
        import rfdetr  # noqa: F401
        import ultralytics  # noqa: F401
    except ImportError as error:
        raise SystemExit(f"Missing inference dependency: {error}. Run `uv sync --locked` first.") from error

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    for folder in ("original", *MODEL_NAMES):
        (output_dir / folder).mkdir()
    results = {"weights": {"rfdetr": str(args.rfdetr_weights), "yolo": str(args.yolo_weights)},
               "input_dir": str(args.input_dir), "confidence": args.confidence,
               "resolution": RESOLUTION, "device": device, "images": []}
    for index, path in enumerate(tqdm(paths, desc="Saving originals", unit="image")):
        filename = f"{index:06d}.jpg"
        read_image(path).save(output_dir / "original" / filename, quality=95)
        results["images"].append({"name": path.relative_to(args.input_dir).as_posix(),
                                  "original": f"original/{filename}"})

    # Load one model at a time so both checkpoints need not fit in GPU memory together.
    for kind, name in MODEL_NAMES.items():
        print(f"{name}: {results['weights'][kind]}\nDevice: {device}", flush=True)
        model = load_model(kind, results["weights"][kind], device)
        for index, path in enumerate(tqdm(paths, desc=name, unit="image")):
            image = read_image(path)
            with torch.inference_mode():
                detections = predict(model, kind, image, args.confidence, device)
            relative = f"{kind}/{index:06d}.jpg"
            annotate(image, detections, COLORS[kind]).save(output_dir / relative, quality=95)
            results["images"][index][kind] = {"image": relative, "detections": detections}
        del model
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    write_results(output_dir, results)
    print(f"Saved predictions and viewer: {output_dir}", flush=True)
    return output_dir


def serve_viewer(results_dir, port):
    handler = partial(SimpleHTTPRequestHandler, directory=str(results_dir))
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        print(f"Viewer: http://localhost:{server.server_port}\nPress Ctrl+C to stop.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rfdetr-weights", type=Path, default=RFDETR_WEIGHTS)
    parser.add_argument("--yolo-weights", type=Path, default=YOLO_WEIGHTS)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR, help="Image folder, searched recursively.")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "out/model-comparison" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f"),
                        help="New folder for images, predictions.json, and index.html.")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-serve", action="store_true", help="Save results without starting the HTTP viewer.")
    parser.add_argument("--results-dir", type=Path, help="Reopen saved results without loading either model.")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535.")
    if args.results_dir:
        results_dir = args.results_dir.expanduser().resolve()
        if not (results_dir / "predictions.json").is_file():
            parser.error(f"No saved predictions.json in {results_dir}")
        write_results(results_dir)
    else:
        if not 0 <= args.confidence <= 1:
            parser.error("--confidence must be between 0 and 1.")
        for field in ("rfdetr_weights", "yolo_weights", "input_dir", "output_dir"):
            setattr(args, field, getattr(args, field).expanduser().resolve())
        for weights in (args.rfdetr_weights, args.yolo_weights):
            if not weights.is_file():
                parser.error(f"Checkpoint not found: {weights}")
        if not args.input_dir.is_dir():
            parser.error(f"Image folder not found: {args.input_dir}")
        if args.output_dir.exists():
            parser.error(f"Output folder already exists; choose a new folder: {args.output_dir}")
        if args.output_dir == args.input_dir or args.input_dir in args.output_dir.parents:
            parser.error("Output folder must be outside the input image folder.")
        paths = sorted(path for path in args.input_dir.rglob("*")
                       if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if not paths:
            parser.error(f"No supported images found in {args.input_dir}")
        results_dir = run_inference(args, paths)
    if args.no_serve:
        print(f"Open in a browser: {(results_dir / 'index.html').as_uri()}")
    else:
        serve_viewer(results_dir, args.port)


if __name__ == "__main__":
    main()
