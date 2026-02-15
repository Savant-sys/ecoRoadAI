# EcoRoad AI — Pipeline overview

The analysis pipeline lives in **`pipeline/ecoroad.py`**. It can be run from the Flask app (upload → run) or from the command line.

## Flow

1. **Input** — Video (or image) path. The app passes the uploaded file path and an optional `--output-id`.
2. **YOLO** — Ultralytics YOLO runs object detection (vehicles, pedestrians, traffic lights, signs, etc.). On GPU, video is processed in batches for speed.
3. **Per-frame** — For each frame: detection boxes, optional optical-flow speed estimate, and drawn annotations (boxes + speed overlay).
4. **Aggregation** — Detection counts are averaged over the clip and used for scene classification and CO₂/fuel estimates.
5. **Analytics** — Scene type, risk level, eco score, driving style, CO₂ by phase, tips, key moments, trends, trip summary, and savings what-if are computed and written to a JSON summary.
6. **Output** — Annotated video (MP4) and a summary JSON (and optional segment outputs in parallel mode).

## Main entry

- **`main()`** — Parses CLI args (`source`, `--segment-out`, `--output-id`), loads the YOLO model, runs inference, then builds the summary and writes outputs. For video on CUDA it uses batched inference; otherwise standard Ultralytics predict.

## Key pieces

| Area | What it does |
|------|----------------|
| **Detection** | `summarize_detections()`, `aggregate_counts_from_frames()` — per-frame and clip-level counts |
| **Scene** | `classify_scene()` — highway / urban / intersection / residential from counts |
| **Speed** | `SpeedEstimator` (optical flow) and `_assumed_mph()` (count-based heuristic) |
| **CO₂ / fuel** | `estimate_co2()` — g/km, fuel penalty %, comparison text |
| **Driving style** | `analyze_driving_style()` — chunks, smoothness, eco score over time |
| **CO₂ by phase** | `analyze_co2_phases()` — segments with high/medium/low CO₂ |
| **Trip & trends** | Trip summary (distance, best/worst segments), trend snapshot, history from `TRIP_HISTORY_PATH` |
| **Recommendations** | Risk level, tips, skill focus, driver assessment from counts and analytics |

## Optional: parallel video

Set **`ECOROAD_PARALLEL_VIDEO=N`** (N > 1) to split the video into N segments, run N processes, then merge summaries and annotated video. Uses `--segment-out` per segment.

## Env (same as README)

- `ECOROAD_MODEL_PATH` — YOLO weights
- `ECOROAD_CONF` — detection confidence
- `ECOROAD_DEVICE` — `cuda` or `cpu`
- `ECOROAD_BATCH_SIZE` — batch size for batched video inference (default 32)
