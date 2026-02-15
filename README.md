# EcoRoad AI

**Drive smarter. Drive greener.** — SF Hacks 2026

EcoRoad AI analyzes dashcam (or any road) video with computer vision and gives you **eco and safety insights** — no car sensors or GPS required. Upload a clip, get an annotated video plus perception counts, scene type, risk level, eco score, CO₂ and fuel estimates, driving style, and actionable tips.

---

## What it does

- **Upload** a short video (dashcam, phone, or any road clip; MP4, MOV, WebM).
- **AI runs** object detection (YOLO) and scene analysis: vehicles, pedestrians, traffic lights, signs, optical-flow speed estimate.
- **You get**:
  - Annotated video with detection boxes and a timeline of alert moments
  - **Perception** — counts (cars, people, lights, signs)
  - **Decision** — scene type, risk level, eco score (0–100), CO₂ g/km (or g/mi)
  - **Recommendations** — tips to drive smoother and use less fuel
  - **Trip analytics** — driving style, fuel & cost (L/gal, km/mi), CO₂ by phase, action plan, key moments, trends, savings what‑if, trip summary

Use the **km / mi** toggle in the nav to switch metric and imperial everywhere (fuel, distance, CO₂). Use **Clear uploads & results** to delete stored videos and outputs. **Gallery** shows all detection videos in one view.

---

## Run the app

**Requirements:** Python 3.10+, pip. GPU optional (faster inference with CUDA).

```bash
cd ecoRoadAI
python -m venv .venv
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

**Mac / Linux:**
```bash
.venv/bin/pip install -r requirements.txt
./run.sh
```

Then open **http://localhost:5050**. Default port is 5050 (to avoid conflict with macOS AirPlay on 5000). To use another port:

- **Windows:** `set PORT=8080 && python app.py`
- **Mac/Linux:** `PORT=8080 ./run.sh`

Upload a video and wait for analysis (~30–60 s depending on length and device). Click **ⓘ** icons for short explanations. **Download video** and **Download results (JSON)** save the annotated clip and full run data.

**Try with sample:** Put a short video at `samples/sample.mp4` (or use clips in `bdd100k/`) and a **Try with sample video** button appears so you can run without uploading.

---

## Tech

| Layer   | Stack |
|--------|--------|
| Backend | Flask |
| Vision  | PyTorch + Ultralytics YOLO (object detection) |
| Pipeline| `pipeline/ecoroad.py` — detection, scene classification, eco/risk rules, CO₂ and fuel estimates, trip summary, trends |

The app uses a **pre-trained** YOLO model (`yolov8n.pt`). Optional fine-tuning on BDD100K: see `scripts/train_bdd100k.py`.

**Optional env:**

- `ECOROAD_MODEL_PATH` — path to weights (default: `yolov8n.pt`)
- `ECOROAD_CONF` — detection confidence 0–1 (default: `0.25`)
- `ECOROAD_DEVICE` — `cuda` or `cpu`

---

## Documentation

- **README** (this file) — overview, run instructions, tech, project layout.
- **docs/** — [Setup](docs/SETUP.md), [Pipeline overview](docs/PIPELINE.md).

---

## Project layout

| Path | Purpose |
|------|--------|
| `app.py` | Flask app entry (port 5050 by default) |
| `run.sh` | Run app with venv on Mac/Linux |
| `app/` | Routes, config, utils (e.g. annotated video with Range support for seeking) |
| `pipeline/ecoroad.py` | Detection + eco/safety + trip pipeline |
| `templates/` | `index.html` (main UI), `gallery.html` |
| `scripts/` | `train_bdd100k.py`, `restore_og_model.py` |
| `videos/` | `uploads/`, `output/`, `analytics/` (created at runtime; cleared only via **Clear uploads & results**) |
| `samples/` | Put `sample.mp4` here for **Try with sample** |
| `docs/` | [SETUP.md](docs/SETUP.md), [PIPELINE.md](docs/PIPELINE.md) |

---

## Notes

- **Distance and speed** are estimated from the video (scene + optical flow), not GPS. Short clips (e.g. 40 s) often show ~0.3–0.5 mi.
- **Trip history and trends** are stored under `videos/analytics/` and persist until you click **Clear uploads & results**.

---

## License

MIT License — see [LICENSE](LICENSE).
