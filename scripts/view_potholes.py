"""Browse local and selected HRP4K images with pothole boxes."""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

# Reuse the training dataset loaders so both scripts select the same images.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pothole_data import (
    DATA_DIR,
    ROOT,
    load_hrp4k_samples,
    load_local_samples,
    split_local_samples,
)


VIEWER_HTML = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pothole viewer</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; padding: 28px; background: #f4f5f7; color: #192434;
         font: 16px system-ui, sans-serif; }
  header { max-width: 1500px; margin: auto auto 24px; }
  h1 { margin: 0 0 8px; font-size: 30px; }
  p { color: #526176; }
  .controls { display: flex; gap: 12px; flex-wrap: wrap; }
  input, select, button { font: inherit; padding: 10px 14px; border: 1px solid #bbc4cf;
                          border-radius: 8px; background: white; }
  input { min-width: 220px; flex: 1; }
  #grid { max-width: 1500px; margin: auto; display: grid; gap: 18px;
          grid-template-columns: repeat(auto-fill, minmax(min(100%, 310px), 1fr)); }
  .card { padding: 0; text-align: left; overflow: hidden; cursor: zoom-in; }
  .card img { width: 100%; aspect-ratio: 16/9; object-fit: contain; display: block;
              background: #101820; }
  .caption { padding: 12px; overflow-wrap: anywhere; }
  .caption small { display: block; margin-top: 5px; color: #526176; }
  dialog { border: 0; border-radius: 12px; padding: 16px; max-width: 96vw;
           max-height: 96vh; }
  dialog::backdrop { background: #101820dd; }
  dialog img { display: block; max-width: 90vw; max-height: 77vh; object-fit: contain; }
  dialog header { display: flex; gap: 16px; align-items: center;
                  justify-content: space-between; margin-bottom: 12px; }
</style>
<header>
  <h1>Pothole viewer</h1>
  <p id="status" role="status">Loading images...</p>
  <div class="controls">
    <input id="search" type="search" placeholder="Search filenames" aria-label="Search filenames">
    <select id="dataset" aria-label="Dataset">
      <option value="all">All datasets</option>
      <option value="local">My images</option>
      <option value="hrp4k">HRP4K (2,000 images)</option>
    </select>
    <select id="split" aria-label="Training split">
      <option value="all">Training + validation</option>
      <option value="train">Training images</option>
      <option value="valid">Validation images</option>
    </select>
    <select id="source" aria-label="Box source">
      <option value="annotations">Dataset annotations</option>
      <option value="predictions">Saved model predictions</option>
      <option value="original">Original images (no boxes)</option>
    </select>
  </div>
</header>
<main id="grid"></main>
<dialog id="preview" aria-labelledby="preview-name">
  <header>
    <strong id="preview-name"></strong>
    <div><button id="previous" aria-label="Previous image">← Previous</button>
    <button id="next" aria-label="Next image">Next →</button>
    <button id="close">Close</button></div>
  </header>
  <img id="preview-image" alt="">
</dialog>
<script>
  const grid = document.querySelector('#grid'), source = document.querySelector('#source');
  const search = document.querySelector('#search'), status = document.querySelector('#status');
  const dataset = document.querySelector('#dataset'), split = document.querySelector('#split');
  const preview = document.querySelector('#preview');
  const previous = document.querySelector('#previous'), next = document.querySelector('#next');
  let images = [], shown = [], currentIndex = 0;
  function openImage(index) {
    if (!shown.length) return;
    currentIndex = Math.max(0, Math.min(index, shown.length - 1));
    const item = shown[currentIndex];
    document.querySelector('#preview-name').textContent =
      `${currentIndex + 1} / ${shown.length} — ${item.name}`;
    const full = document.querySelector('#preview-image');
    full.src = `/image/${item.id}?source=${source.value}&full=1`; full.alt = item.name;
    previous.disabled = currentIndex === 0; next.disabled = currentIndex === shown.length - 1;
    if (!preview.open) preview.showModal();
  }
  function render() {
    if (preview.open) preview.close();
    grid.replaceChildren();
    shown = images.filter(item => item.name.toLowerCase().includes(search.value.toLowerCase())
      && (dataset.value === 'all' || item.dataset === dataset.value)
      && (split.value === 'all' || item.split === split.value)
      && (source.value !== 'predictions' || item.prediction));
    const descriptions = {annotations: 'Red boxes are dataset labels.',
      predictions: 'Boxes and confidence scores are from saved model predictions.',
      original: 'Images without boxes.'};
    status.textContent = `${shown.length} / ${images.length} images. ${descriptions[source.value]} Click to enlarge; use ← → to browse and Esc to close.`;
    if (!shown.length) {
      grid.textContent = source.value === 'predictions'
        ? 'No matching predictions. Run the inference script or select another predictions folder.'
        : 'No matching images.';
    }
    shown.forEach((item, index) => {
      const card = document.createElement('button');
      card.className = 'card';
      const image = document.createElement('img');
      const url = `/image/${item.id}?source=${source.value}`;
      image.src = url; image.alt = item.name; image.loading = 'lazy';
      const caption = document.createElement('div'); caption.className = 'caption';
      caption.textContent = item.name;
      const detail = document.createElement('small');
      const label = source.value === 'annotations' ? `${item.count} annotated potholes`
        : source.options[source.selectedIndex].text;
      detail.textContent = `${item.dataset === 'local' ? 'My images' : 'HRP4K'} · ${item.split === 'train' ? 'Training' : 'Validation'} · ${label}`;
      caption.append(detail); card.append(image, caption);
      card.onclick = () => openImage(index);
      grid.append(card);
    });
  }
  search.oninput = render; source.onchange = render;
  dataset.onchange = render; split.onchange = render;
  previous.onclick = () => openImage(currentIndex - 1);
  next.onclick = () => openImage(currentIndex + 1);
  document.querySelector('#close').onclick = () => preview.close();
  document.addEventListener('keydown', event => {
    if (event.target.matches('input, textarea, select') || event.target.isContentEditable) return;
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    event.preventDefault();
    openImage(preview.open ? currentIndex + (event.key === 'ArrowRight' ? 1 : -1) : 0);
  });
  fetch('/api/images').then(response => {
    if (!response.ok) throw Error('Could not load images');
    return response.json();
  }).then(data => { images = data; render(); }).catch(error => { status.textContent = error.message; });
</script>
</html>"""


def serve_viewer(port, predictions_dir):
    """Show dataset boxes or saved predictions without loading the model."""
    from PIL import Image, ImageDraw

    local_samples = load_local_samples()
    _, valid_samples = split_local_samples(local_samples)
    valid_paths = {path for path, _ in valid_samples}
    # Use the same seeded HRP4K selection as fine-tuning, including its box labels.
    samples = local_samples + load_hrp4k_samples()
    predictions = [predictions_dir / path.relative_to(DATA_DIR / "processed_images")
                   for path, _ in local_samples] + [None] * (len(samples) - len(local_samples))
    entries = [{"id": index, "name": path.name, "count": len(boxes),
                "dataset": "local" if index < len(local_samples) else "hrp4k",
                "split": "valid" if path in valid_paths else "train",
                "prediction": predictions[index] is not None and predictions[index].is_file()}
               for index, (path, boxes) in enumerate(samples)]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request = urlsplit(self.path)
            if request.path == "/":
                body, content_type = VIEWER_HTML.encode(), "text/html; charset=utf-8"
            elif request.path == "/api/images":
                body, content_type = json.dumps(entries).encode(), "application/json"
            elif request.path.startswith("/image/"):
                try:
                    index = int(request.path.removeprefix("/image/"))
                    if not 0 <= index < len(samples):
                        raise ValueError("Unknown image")
                    path, boxes = samples[index]
                    query = parse_qs(request.query)
                    source = query.get("source", ["annotations"])[0]
                    if source not in {"annotations", "predictions", "original"}:
                        raise ValueError("Unknown source")
                    if source == "predictions":
                        path = predictions[index]
                        if path is None:
                            raise ValueError("No saved prediction for this image")
                    with Image.open(path) as original:
                        image = original.convert("RGB")
                    width, height = image.size
                    size = 1920 if query.get("full") == ["1"] else 640
                    image.thumbnail((size, size))
                    if source == "annotations":
                        draw = ImageDraw.Draw(image)
                        sx, sy = image.width / width, image.height / height
                        for x, y, w, h in boxes:
                            draw.rectangle([x * sx, y * sy, (x + w) * sx, (y + h) * sy],
                                           outline="#ff3535", width=3)
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
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Pothole viewer: http://localhost:{server.server_port} ({len(samples)} images)", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Browse the pothole fine-tuning datasets.")
    parser.add_argument("--port", type=int, default=8000, help="Viewer port (default: 8000).")
    parser.add_argument("--predictions-dir", type=Path,
                        default=ROOT / "out" / "hrp4k-rfdetr-small",
                        help="Folder of saved annotated model predictions.")
    args = parser.parse_args()
    serve_viewer(args.port, args.predictions_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
