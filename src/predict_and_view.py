"""Run a saved pothole checkpoint on local images and serve a localhost viewer."""

import argparse
import gc
import json
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "out/hrp4k-rfdetr-small-finetuned/2026-10-25_19-56-17_610318/checkpoint_best_total.pth"
INPUT_DIR = ROOT / "data/processed_images"
CONFIDENCE = 0.30
RESOLUTION = 896
PORT = 8000
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


PAGE = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pothole predictions</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; padding: 24px; background: #f4f5f7; color: #192434;
         font: 16px system-ui, sans-serif; }
  header, main { max-width: 1500px; margin: auto; }
  h1 { margin: 0 0 8px; }
  p { color: #526176; }
  .controls { display: flex; gap: 12px; flex-wrap: wrap; margin: 20px 0; }
  input, select, button { font: inherit; padding: 10px 14px; background: white;
                          border: 1px solid #bbc4cf; border-radius: 8px; }
  input { flex: 1; min-width: 220px; }
  #grid { display: grid; gap: 18px;
          grid-template-columns: repeat(auto-fill, minmax(min(100%, 310px), 1fr)); }
  .card { padding: 0; overflow: hidden; text-align: left; cursor: zoom-in; }
  .card img { display: block; width: 100%; aspect-ratio: 16/9; object-fit: contain;
              background: #101820; }
  .caption { padding: 12px; overflow-wrap: anywhere; }
  small { display: block; color: #526176; margin-top: 5px; }
  dialog { border: none; border-radius: 12px; padding: 16px; max-width: 96vw; max-height: 96vh; }
  dialog::backdrop { background: #101820dd; }
  dialog header { display: flex; align-items: center; justify-content: space-between;
                  gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }
  dialog img { display: block; max-width: 90vw; max-height: 77vh; object-fit: contain; }
</style>
<header>
  <h1>Pothole predictions</h1>
  <p id="model"></p>
  <div class="controls">
    <input id="search" type="search" placeholder="Search filenames" aria-label="Search filenames">
    <select id="filter" aria-label="Filter detections">
      <option value="all">All images</option>
      <option value="positive">With detections</option>
      <option value="negative">Without detections</option>
    </select>
    <select id="source" aria-label="Image display">
      <option value="predictions">Predicted boxes</option>
      <option value="original">Original images</option>
    </select>
  </div>
  <p id="status" role="status">Loading predictions...</p>
</header>
<main id="grid"></main>
<dialog id="preview" aria-labelledby="preview-title">
  <header>
    <strong id="preview-title"></strong>
    <div><button id="previous">← Previous</button>
      <button id="next">Next →</button><button id="close">Close</button></div>
  </header>
  <img id="full-image" alt="">
</dialog>
<script>
  const grid = document.querySelector('#grid'), search = document.querySelector('#search');
  const filter = document.querySelector('#filter'), source = document.querySelector('#source');
  const preview = document.querySelector('#preview'), status = document.querySelector('#status');
  const previous = document.querySelector('#previous'), next = document.querySelector('#next');
  let images = [], shown = [], current = 0;
  function imageURL(item, full = false) {
    return `/image/${item.id}?source=${source.value}&full=${full ? 1 : 0}`;
  }
  function openImage(index) {
    if (!shown.length) return;
    current = Math.max(0, Math.min(index, shown.length - 1));
    const item = shown[current], image = document.querySelector('#full-image');
    document.querySelector('#preview-title').textContent =
      `${current + 1} / ${shown.length} · ${item.name} · ${item.count} detections`;
    image.src = imageURL(item, true); image.alt = item.name;
    previous.disabled = current === 0; next.disabled = current === shown.length - 1;
    if (!preview.open) preview.showModal();
  }
  function render() {
    if (preview.open) preview.close();
    shown = images.filter(item => item.name.toLowerCase().includes(search.value.toLowerCase())
      && (filter.value === 'all' || (filter.value === 'positive' ? item.count > 0 : item.count === 0)));
    grid.replaceChildren();
    status.textContent = `${shown.length} / ${images.length} images. Click to enlarge; ← → to browse; Esc to close.`;
    if (!shown.length) grid.textContent = 'No matching images.';
    shown.forEach((item, index) => {
      const card = document.createElement('button'); card.className = 'card';
      const image = document.createElement('img');
      image.src = imageURL(item); image.alt = item.name; image.loading = 'lazy';
      const caption = document.createElement('div'); caption.className = 'caption';
      caption.textContent = item.name;
      const count = document.createElement('small'); count.textContent = `${item.count} predicted potholes`;
      caption.append(count); card.append(image, caption);
      card.onclick = () => openImage(index); grid.append(card);
    });
  }
  search.oninput = render; filter.onchange = render; source.onchange = render;
  previous.onclick = () => openImage(current - 1); next.onclick = () => openImage(current + 1);
  document.querySelector('#close').onclick = () => preview.close();
  document.addEventListener('keydown', event => {
    if (event.target.matches('input, textarea, select') || event.target.isContentEditable) return;
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    event.preventDefault();
    openImage(preview.open ? current + (event.key === 'ArrowRight' ? 1 : -1) : 0);
  });
  fetch('/api/results').then(response => {
    if (!response.ok) throw Error('Could not load results');
    return response.json();
  }).then(results => {
    images = results.images;
    const count = images.reduce((total, item) => total + item.count, 0);
    const name = results.weights.split('/').slice(-2).join('/');
    document.querySelector('#model').textContent =
      `${name} · confidence ≥ ${results.confidence} · ${count} predicted potholes`;
    render();
  }).catch(error => { status.textContent = error.message; });
</script>
</html>"""


def run_inference(weights, input_dir, confidence):
    import torch
    from rfdetr import RFDETRSmall

    paths = sorted(path for path in input_dir.rglob("*")
                   if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise SystemExit(f"No supported images found in {input_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Checkpoint: {weights}\nImages: {len(paths)}\nDevice: {device}")
    model = RFDETRSmall.from_checkpoint(weights, device=device, resolution=RESOLUTION)
    if "pothole" not in [name.lower() for name in model.class_names]:
        raise ValueError(f"The checkpoint has no pothole class: {model.class_names}")

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output_dir = weights.parent / "predictions" / f"{weights.stem}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    font = ImageFont.load_default(size=24)
    results = {"weights": str(weights), "input_dir": str(input_dir),
               "confidence": confidence, "resolution": RESOLUTION, "images": []}
    for index, path in enumerate(tqdm(paths, desc="Predicting potholes", unit="image")):
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
        with torch.inference_mode():
            detections = model.predict(image, threshold=confidence)
        boxes, scores = [], []
        draw = ImageDraw.Draw(image)
        for box, score, name in zip(detections.xyxy, detections.confidence,
                                    detections.data["class_name"]):
            if str(name).lower() != "pothole":
                continue
            x1, y1, x2, y2 = map(float, box)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=max(3, image.width // 500))
            label = f"Pothole {score:.2f}"
            left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
            label_y = max(0, y1 - (bottom - top) - 10)
            draw.rectangle([x1, label_y, x1 + right - left + 12, label_y + bottom - top + 10], fill="red")
            draw.text((x1 + 6 - left, label_y + 5 - top), label, font=font, fill="white")
            boxes.append([x1, y1, x2, y2])
            scores.append(float(score))
        relative = path.relative_to(input_dir)
        destination = output_dir / "images" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, quality=95)
        results["images"].append({"id": index, "name": relative.as_posix(),
                                  "count": len(boxes), "boxes_xyxy": boxes, "scores": scores})

    (output_dir / "predictions.json").write_text(json.dumps(results, indent=2))
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"Predictions saved to: {output_dir}")
    return output_dir


def serve_viewer(results_dir, port):
    results = json.loads((results_dir / "predictions.json").read_text())
    input_dir = Path(results["input_dir"])

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request = urlsplit(self.path)
            if request.path == "/":
                body, content_type = PAGE.encode(), "text/html; charset=utf-8"
            elif request.path == "/api/results":
                body, content_type = json.dumps(results).encode(), "application/json"
            elif request.path.startswith("/image/"):
                try:
                    index = int(request.path.removeprefix("/image/"))
                    if not 0 <= index < len(results["images"]):
                        raise ValueError("Unknown image")
                    item = results["images"][index]
                    query = parse_qs(request.query)
                    root = input_dir if query.get("source") == ["original"] else results_dir / "images"
                    with Image.open(root / item["name"]) as original:
                        image = ImageOps.exif_transpose(original).convert("RGB")
                    size = 1920 if query.get("full") == ["1"] else 640
                    image.thumbnail((size, size))
                    buffer = BytesIO()
                    image.save(buffer, format="JPEG", quality=90)
                    body, content_type = buffer.getvalue(), "image/jpeg"
                except (ValueError, OSError):
                    self.send_error(404, "Image unavailable")
                    return
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"Viewer: http://localhost:{server.server_port}\nPress Ctrl+C to stop.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=WEIGHTS, help="Saved .pth or .ckpt checkpoint.")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR, help="Image folder, searched recursively.")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--results-dir", type=Path, help="Reopen saved predictions without running inference.")
    args = parser.parse_args()
    if args.results_dir:
        results_dir = args.results_dir.expanduser().resolve()
        if not (results_dir / "predictions.json").is_file():
            parser.error(f"No predictions.json in {results_dir}")
    else:
        weights, input_dir = args.weights.expanduser().resolve(), args.input_dir.expanduser().resolve()
        if not weights.is_file():
            parser.error(f"Checkpoint not found: {weights}")
        if not input_dir.is_dir():
            parser.error(f"Image folder not found: {input_dir}")
        if not 0 <= args.confidence <= 1:
            parser.error("Confidence must be between 0 and 1.")
        if input_dir == weights.parent or input_dir in weights.parents:
            parser.error("Keep the checkpoint outside the input image folder to avoid including saved predictions.")
        results_dir = run_inference(weights, input_dir, args.confidence)
    serve_viewer(results_dir, args.port)


if __name__ == "__main__":
    main()
