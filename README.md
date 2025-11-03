# CatFaceID

Open‑set face identification for cats. Includes training hooks, enrollment, live identification, and a simple web UI.

## Features

- Detect cat heads with a YOLOv8 detector.
- Enroll pets from folders or via webcam; auto‑align via CAT landmarks when available.
- Identify single images, folders, or live webcam feed.
- Open‑set UNKNOWN handling with similarity threshold + margin rules.
- Quality filter (blur/size) to skip low‑quality frames.
- Threshold sweeper (FAR/FRR) to choose good reject/margin settings.

## Local Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# (Optional) train your detector → writes to runs/cat_head
yolo detect train data=data/pet_head.yaml model=yolov8n.pt imgsz=640 epochs=60 batch=16 project=runs name=cat_head

# Build the gallery from folders under data/faceid/<pet_id>/
python3 scripts/enroll_gallery.py

# Start API (FastAPI/uvicorn) on :9000
python3 api/server.py

# Serve the frontend on :8000 (in another shell)
cd web && python3 -m http.server 8000
```

Open http://127.0.0.1:8000 to use the UI. The backend runs on http://127.0.0.1:9000.

## Frontend API Base

The web UI calls the backend using a base URL in `web/app.js`:

```js
const API = "http://127.0.0.1:9000";
```

Change this if you deploy the API elsewhere.

## Enroll and Identify

- Folders → Put images in `data/faceid/<pet_id>/`, then run:
  - `python3 scripts/enroll_gallery.py` (writes `gallery/` vectors/labels/meta)
- Camera capture (CLI):
  - `python3 scripts/cam_capture_enroll.py` to save crops into `data/faceid/<pet_id>/` and auto‑rebuild the gallery.
  - `python3 scripts/cam_identify.py` for live identification (press S to snapshot overlays).
- Web UI:
  - Enroll tab lets you capture/select images, add to a set, and upload to the API (`/enroll`) to rebuild the gallery.
  - Identify tab lets you upload or capture, then calls the API (`/identify`).

## Data Locations

- `data/faceid/`    — per‑pet source images (per folder).
- `gallery/`        — enrolled vectors/labels/meta (used for identification).
- `runs/`           — YOLO training/exports and snapshots (e.g., `runs/cat_head`).

## Unknown Handling

Open‑set rules are enforced during identification:

- Reject if top‑1 similarity `< reject` (default 0.75), or if `(top1 − top2) < margin` (default 0.05).
- Tune with `scripts/sweep_threshold.py`, which reports FAR/FRR across sweeps and recommends settings.

## Deployment Notes

- API: `api/server.py` (FastAPI). Add your frontend origin to CORS in `api/server.py`.
- Frontend: serve `web/` statically (e.g., Netlify). Update `web/app.js` API base.
- One simple setup: Deploy API on Render/Fly/Heroku, serve UI on Netlify/Vercel.
- Tunnels for local demos: Cloudflare Tunnel or ngrok pointing frontend and backend ports to public URLs.

## Extras

- Optional FAISS nearest‑neighbour search: run identify with `--faiss` for faster lookups on larger galleries.
- macOS camera tips: grant camera permission to your terminal/VS Code; fallback backend (AVFoundation) is enabled in camera scripts.
