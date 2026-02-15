# EcoRoad AI

**SF Hacks 2026 — Tech for a Greener Tomorrow**

EcoRoad AI analyzes dashcam (or any road) video with computer vision and gives you **eco and safety insights**: what the AI saw, scene type, risk level, eco score, CO₂ estimates, trip summary, and actionable tips — no car sensors required.

---

## What it does

- **Upload** a short video (dashcam, phone, or any road clip).
- **AI runs** object detection (YOLO) and analyzes the scene: vehicles, pedestrians, traffic lights, signs.
- **You get** an annotated video, perception counts, scene type (highway/urban/intersection/residential), risk level, eco score (0–100), CO₂ g/km estimate, fuel/cost estimates, driving-style summary, and tips to drive smoother and use less fuel.

Use it to **save money**, **prove safe driving**, or **review fleet/family trips**.

---

## Run the app

**Requirements:** Python 3.10+, `pip`, (optional) GPU for faster inference.

```bash
git clone <your-repo-url>
cd ecoRoadAI
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./run.sh
```

Then open **http://localhost:5050** (default port 5050 to avoid macOS AirPlay on 5000).  
To use another port: `PORT=8080 ./run.sh`

Upload a video, wait for analysis (~30–60 s depending on length and device), then explore the results. Click the **ⓘ** icons for explanations. Use **km / mi** in the nav to switch metric/imperial; **Download results (JSON)** to export the run.

**Try with sample:** If you add a short video as `samples/sample.mp4` (or have clips in `bdd100k/`), a **Try with sample video** button appears so you can run the pipeline without uploading.

---

## Tech

- **Backend:** Flask
- **Vision:** PyTorch + Ultralytics YOLO (object detection)
- **Pipeline:** `pipeline/ecoroad.py` — detection, scene classification, eco/risk rules, CO₂ and fuel estimates, trip summary

The app uses a **pre-trained** YOLO model (`yolov8n.pt`). You can optionally fine-tune on BDD100K (see `scripts/train_bdd100k.py` and `docs/`).

**Optional env:** `ECOROAD_MODEL_PATH` — path to weights (default: `yolov8n.pt`). `ECOROAD_CONF` — detection confidence 0–1 (default: `0.25`). `ECOROAD_DEVICE` — `cuda` or `cpu`.

---

## Project layout

| Path | Purpose |
|------|--------|
| `app.py` | Flask app entry |
| `run.sh` | Run app with venv (port 5050) |
| `app/` | Routes, config, utils |
| `pipeline/ecoroad.py` | Detection + eco/safety pipeline |
| `templates/index.html` | Single-page UI (upload, loading, results) |
| `scripts/` | `clean.sh` (cache/video cleanup), `train_bdd100k.py` (optional training) |
| `tests/` | Unit and integration tests |
| `docs/` | Result explanations, testing summary |
| `samples/` | Put `sample.mp4` here for **Try with sample** (or use `bdd100k/*.mov`) |

---

## Tests

```bash
.venv/bin/pytest tests/test_app.py -v
.venv/bin/python tests/test_integration_bdd100k.py   # optional; needs bdd100k/*.mov
```

See `tests/README.md` for more options.

---

## License

See repository for license details.
