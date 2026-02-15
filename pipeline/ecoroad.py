import os
import sys

# Use all logical cores for PyTorch/OpenMP (must set before torch/ultralytics load)
_cpu_count = os.cpu_count() or 32
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, str(_cpu_count))

import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import cv2
from ultralytics import YOLO

# PyTorch intra-op/inter-op threads (after torch is loaded by ultralytics)
import torch
torch.set_num_threads(_cpu_count)
if hasattr(torch, "set_num_interop_threads"):
    torch.set_num_interop_threads(min(16, max(1, _cpu_count // 2)))

from app.config import ROOT, OUTPUT_DIR, DETECT_DIR, OUTPUT_PARALLEL_DIR, TRIP_HISTORY_PATH

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".webm", ".mkv")


def _inference_device():
    """Use GPU (CUDA) if available, else CPU. Override with env ECOROAD_DEVICE=cuda|cpu."""
    want = os.environ.get("ECOROAD_DEVICE", "").strip().lower()
    if want in ("cuda", "gpu", "0"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if want == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _reencode_to_web_mp4(src: Path, dst: Path) -> bool:
    """Re-encode video to H.264 MP4 so it plays in browser. Returns True on success."""
    r = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(src),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-movflags", "+faststart", "-pix_fmt", "yuv420p", "-an",
            str(dst),
        ],
        capture_output=True,
        cwd=str(ROOT),
    )
    if r.returncode != 0 or not dst.exists():
        return False
    return True


def _get_video_duration_sec(path):
    """Return duration in seconds via ffprobe."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    if out.returncode != 0:
        return None
    try:
        return float(out.stdout.strip())
    except ValueError:
        return None


def _run_parallel_video(source: str, n: int):
    """Split video into n segments, run n ecoroad processes, merge summaries and annotated video."""
    source = Path(source).resolve()
    if not source.is_file():
        print("Source not found:", source)
        sys.exit(1)
    duration = _get_video_duration_sec(source)
    if duration is None or duration <= 0:
        print("Could not get video duration; run without ECOROAD_PARALLEL_VIDEO.")
        sys.exit(1)
    seg_dur = duration / n
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    par_out = OUTPUT_PARALLEL_DIR
    par_out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(ROOT)) as tmp:
        tmp = Path(tmp)
        # Split with ffmpeg (keyframe-accurate -ss after -i for accuracy)
        for i in range(n):
            start = i * seg_dur
            seg_path = tmp / f"seg_{i}.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-ss", str(start), "-i", str(source),
                    "-t", str(seg_dur), "-c", "copy", str(seg_path),
                ],
                capture_output=True,
                cwd=str(ROOT),
            )
            if not seg_path.is_file():
                print("Segment failed:", seg_path)
                sys.exit(1)
        # Run n ecoroad processes
        procs = []
        for i in range(n):
            seg_path = tmp / f"seg_{i}.mp4"
            seg_out = par_out / str(i)
            seg_out.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            procs.append(
                subprocess.Popen(
                    [sys.executable, "-m", "pipeline.ecoroad", str(seg_path), "--segment-out", str(seg_out)],
                    cwd=str(ROOT),
                    env=env,
                )
            )
        for p in procs:
            p.wait()
        if any(p.returncode != 0 for p in procs):
            print("Some segment processes failed.")
            sys.exit(1)
        # Merge summaries (weighted average by frames)
        total_frames = 0
        weighted = {}
        for i in range(n):
            summary_path = par_out / str(i) / "summary.json"
            if not summary_path.exists():
                continue
            data = json.loads(summary_path.read_text())
            f = data.get("frames_processed", 0)
            total_frames += f
            det = data.get("detections") or {}
            for k, v in det.items():
                if isinstance(v, (int, float)):
                    weighted[k] = weighted.get(k, 0) + v * f
        merged_counts = {}
        if total_frames > 0:
            for k, v in weighted.items():
                merged_counts[k] = round(v / total_frames, 1)
        eco = eco_safety_rules(merged_counts)
        summary = {
            "source": str(source),
            "media_type": "video",
            "frames_processed": total_frames,
            "detections": merged_counts,
            "annotated_media": "annotated.mp4",
            **eco,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        (out_dir / "last_media_type.txt").write_text("video")
        # Concat annotated segments
        list_file = tmp / "concat.txt"
        lines = []
        for i in range(n):
            p = (par_out / str(i) / "annotated.mp4")
            if p.exists():
                path_str = p.resolve().as_posix().replace("'", "'\\''")
                lines.append(f"file '{path_str}'")
        list_file.write_text("\n".join(lines))
        out_mp4 = out_dir / "annotated.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(out_mp4)],
            capture_output=True,
            cwd=str(ROOT),
        )
        if not out_mp4.is_file():
            print("Concat failed; segment outputs left in output_parallel/")
        else:
            print("Saved merged summary and", out_mp4.name)
        # Optionally remove segment outputs to save space
        # for i in range(n): shutil.rmtree(par_out / str(i), ignore_errors=True)


def _run_batched_video(model, source, predict_kw, run_name, device):
    """Process video in batches to keep GPU busy. Returns (results_list, output_dir)."""
    try:
        batch_size = int(os.environ.get("ECOROAD_BATCH_SIZE", "32" if device == "cuda" else "8"))
    except ValueError:
        batch_size = 32 if device == "cuda" else 8
    batch_size = max(1, min(batch_size, 64))

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_name = Path(source).name

    runs_dir = DETECT_DIR
    if run_name:
        out_dir = runs_dir / run_name
    else:
        out_dir = runs_dir / "predict_batch"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / source_name

    # Copy predict_kw but no save, no project/name (we write frames ourselves)
    kw = {k: v for k, v in predict_kw.items() if k not in ("save", "project", "name", "exist_ok")}
    kw["save"] = False
    kw["verbose"] = False

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    results_list = []
    batch = []
    raw_frames = []  # keep original frames for optical flow
    speed_est = SpeedEstimator(fps=fps)

    def _write_batch(pred_results, orig_frames):
        for r, orig in zip(pred_results, orig_frames):
            results_list.append(r)
            img = r.plot()
            if img is not None:
                counts = summarize_detections([r])
                mph = speed_est.update(orig, counts)
                _draw_speed_on_frame(img, mph)
                writer.write(img)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        batch.append(frame)
        raw_frames.append(frame)
        if len(batch) < batch_size:
            continue
        pred = model.predict(source=batch, batch=len(batch), **kw)
        _write_batch(pred, raw_frames)
        batch = []
        raw_frames = []

    if batch:
        pred = model.predict(source=batch, batch=len(batch), **kw)
        _write_batch(pred, raw_frames)

    cap.release()
    writer.release()
    speed_est.finalize()
    return results_list, out_dir, fps, speed_est


def summarize_detections(results):
    """Single frame: results is list of one Result. Video: pass one Result at a time."""
    r = results[0] if isinstance(results, list) else results
    names = r.names
    counts = {}
    if r.boxes is not None and len(r.boxes) > 0:
        for cls_id in r.boxes.cls.tolist():
            name = names[int(cls_id)]
            counts[name] = counts.get(name, 0) + 1
    return counts


def aggregate_counts_from_frames(results_list):
    """Average detection counts per frame (keep 1 decimal so truck/bus etc. don't round to 0)."""
    total = {}
    for r in results_list:
        c = summarize_detections([r])
        for k, v in c.items():
            total[k] = total.get(k, 0) + v
    n = len(results_list)
    if n == 0:
        return {}
    return {k: round(v / n, 1) for k, v in total.items()}


def _assumed_mph(counts):
    """
    Continuous speed estimate (mph) from detection counts.

    Starts from a scene-type base speed, then applies per-object penalties
    so that heavier traffic / more pedestrians = lower speed.
    Based on FHWA speed-flow curves (simplified):
      - Free-flow highway ≈ 65 mph, drops ~3 mph per extra vehicle in view
      - Urban arterial ≈ 35 mph base, drops faster with pedestrians
      - Intersection ≈ 25 mph base (stop-controlled)
      - Residential ≈ 30 mph base
    Heavy vehicles (trucks/buses) penalize more than cars (slower accel, wider).
    """
    scene = classify_scene(counts)
    stype = scene["scene_type"]

    cars = _f(counts, "car") + _f(counts, "motorcycle") + _f(counts, "motor") + _f(counts, "bike")
    heavy = _f(counts, "truck") + _f(counts, "bus")
    c = _norm_counts(counts)
    people = float(c["_people"])
    stops = float(c["_stops"])

    # Base speed by scene type
    if stype == "highway":
        base = 65
    elif stype == "urban":
        base = 35
    elif stype == "intersection":
        base = 25
    else:
        base = 30

    # Vehicle congestion penalty — each extra vehicle in the frame means denser traffic
    total_vehicles = cars + heavy
    if stype == "highway":
        # Highway: expect a few cars; penalize above 3
        base -= max(0, total_vehicles - 3) * 3
        base -= heavy * 4  # trucks/buses slow highway traffic significantly
    else:
        # City/residential: penalize above 1
        base -= max(0, total_vehicles - 1) * 3
        base -= heavy * 3

    # Pedestrian penalty — drivers must slow for foot traffic
    if people >= 3:
        base -= 10
    elif people >= 1:
        base -= 5

    # Traffic control penalty — stop signs / lights mean expected stops
    if stops >= 2:
        base -= 6
    elif stops >= 1:
        base -= 3

    # Clamp to sane range per scene
    if stype == "highway":
        return max(30, min(70, round(base)))
    elif stype == "intersection":
        return max(10, min(30, round(base)))
    else:
        return max(10, min(45, round(base)))


class SpeedEstimator:
    """
    Estimate ego-vehicle speed from optical flow (frame-to-frame pixel motion).

    Uses dense optical flow (Farneback) on the bottom 40% of the frame (road surface)
    to measure how fast the scene is moving. The median flow magnitude is converted to
    mph via a calibration factor, then smoothed with an exponential moving average so
    the number changes gradually instead of jumping every frame.

    For the first frame (no previous frame), falls back to the detection-based heuristic.
    """

    # Calibration: maps p75 optical-flow magnitude (pixels/frame) to mph.
    # At 0.5 scale on 640px dashcam, 5px/frame real shift ≈ 2.5px flow ≈ 30 mph.
    # So ~12 mph per pixel of flow at 30 fps. Tuned on synthetic + real dashcam.
    FLOW_TO_MPH = 12.0
    MAX_MPH = 75
    EMA_ALPHA = 0.12   # smoothing: ~8-frame half-life at 30fps (responsive but stable)
    NOISE_FLOOR = 0.3  # below this flow magnitude, assume stopped (camera vibration)
    IDLE_THRESHOLD_MPH = 2  # below this speed = idling
    HARD_BRAKE_MPH_DROP = 10  # speed drop per second that counts as hard braking

    def __init__(self, fps=30.0):
        self.fps = fps
        self.prev_gray = None
        self.smooth_mph = None
        self._scale = 0.5
        self._frame_idx = 0
        # Idle tracking
        self._idle_start = None
        self.idle_events = []       # [{"start_frame": int, "duration_sec": float}, ...]
        self.total_idle_frames = 0
        # Hard braking tracking
        self._speed_history = []    # rolling window of (frame_idx, mph)
        self.hard_brake_events = [] # [{"frame": int, "speed_drop_mph": float}, ...]

    def update(self, frame_bgr, counts=None):
        """Feed a BGR frame, return smoothed speed in mph."""
        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (int(w * self._scale), int(h * self._scale)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # Use only bottom 40% of frame (road surface, most motion)
        sh = gray.shape[0]
        roi = gray[int(sh * 0.6):, :]

        if self.prev_gray is not None and self.prev_gray.shape == roi.shape:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, roi, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            # Use 75th percentile instead of median — more sensitive to dominant motion,
            # less affected by static regions (parked cars, sky leaking into ROI)
            p75 = float(np.percentile(mag, 75))
            # Clamp to 0 below noise floor (camera vibration when parked)
            if p75 < self.NOISE_FLOOR:
                raw_mph = 0.0
            else:
                # Scale by fps ratio (calibrated at 30 fps)
                raw_mph = p75 * self.FLOW_TO_MPH * (self.fps / 30.0)
                raw_mph = min(raw_mph, self.MAX_MPH)
        else:
            # First frame: fall back to detection heuristic
            raw_mph = float(_assumed_mph(counts or {}))

        self.prev_gray = roi

        # Exponential moving average for smooth transitions
        if self.smooth_mph is None:
            self.smooth_mph = raw_mph
        else:
            self.smooth_mph += self.EMA_ALPHA * (raw_mph - self.smooth_mph)

        speed = max(0, round(self.smooth_mph))

        # --- Idle detection ---
        if speed < self.IDLE_THRESHOLD_MPH:
            self.total_idle_frames += 1
            if self._idle_start is None:
                self._idle_start = self._frame_idx
        else:
            if self._idle_start is not None:
                idle_frames = self._frame_idx - self._idle_start
                if idle_frames >= 5 * self.fps:  # 5+ seconds = idle event
                    self.idle_events.append({
                        "start_frame": self._idle_start,
                        "duration_sec": round(idle_frames / self.fps, 1),
                    })
                self._idle_start = None

        # --- Hard braking detection ---
        self._speed_history.append((self._frame_idx, speed))
        window = int(self.fps)  # 1-second rolling window
        # trim old entries
        self._speed_history = [(f, s) for f, s in self._speed_history if self._frame_idx - f <= window]
        if len(self._speed_history) >= 2:
            max_speed_in_window = max(s for _, s in self._speed_history)
            drop = max_speed_in_window - speed
            if drop >= self.HARD_BRAKE_MPH_DROP:
                # only record if not duplicate (last event was >1s ago)
                if not self.hard_brake_events or (self._frame_idx - self.hard_brake_events[-1]["frame"]) > window:
                    self.hard_brake_events.append({
                        "frame": self._frame_idx,
                        "speed_drop_mph": round(drop, 1),
                    })

        self._frame_idx += 1
        return speed

    def finalize(self):
        """Call after last frame to close any open idle streak."""
        if self._idle_start is not None:
            idle_frames = self._frame_idx - self._idle_start
            if idle_frames >= 5 * self.fps:
                self.idle_events.append({
                    "start_frame": self._idle_start,
                    "duration_sec": round(idle_frames / self.fps, 1),
                })
            self._idle_start = None

    def get_idle_stats(self, total_frames=None):
        """Return idle summary."""
        total = total_frames or self._frame_idx or 1
        return {
            "idle_events": self.idle_events,
            "total_idle_sec": round(self.total_idle_frames / self.fps, 1),
            "idle_pct": round(self.total_idle_frames / total * 100, 1),
        }

    def get_brake_stats(self):
        """Return hard braking summary."""
        return {
            "hard_brake_events": self.hard_brake_events,
            "hard_brake_count": len(self.hard_brake_events),
        }


def _draw_speed_on_frame(img, mph):
    """Draw speed (mph) on the frame; img is BGR numpy array from result.plot()."""
    text = "~%d mph" % mph
    x, y = 24, 56
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.1
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + 4), (0, 0, 0), -1)
    cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + 4), (255, 255, 255), 1)
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def detect_conditions(frame_bgr):
    """
    Detect lighting/weather conditions from a single frame.
    Returns: "night", "overcast", or "day" based on brightness and contrast.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    std_brightness = float(np.std(gray))
    if mean_brightness < 60:
        return "night"
    if mean_brightness < 100 and std_brightness < 35:
        return "overcast"
    return "day"


def aggregate_conditions(results_list, fps):
    """Sample frames at ~1 fps and return condition percentages."""
    if not results_list:
        return {"day_pct": 100, "night_pct": 0, "overcast_pct": 0, "primary": "day"}
    step = max(1, int(fps))
    counts = {"day": 0, "night": 0, "overcast": 0}
    for i in range(0, len(results_list), step):
        r = results_list[i]
        if hasattr(r, "orig_img") and r.orig_img is not None:
            cond = detect_conditions(r.orig_img)
            counts[cond] += 1
    total = sum(counts.values()) or 1
    result = {
        "day_pct": round(counts["day"] / total * 100, 1),
        "night_pct": round(counts["night"] / total * 100, 1),
        "overcast_pct": round(counts["overcast"] / total * 100, 1),
    }
    result["primary"] = max(counts, key=counts.get)
    return result


def _physics_tip(eco, counts):
    """
    Speed/distance tip using simple physics. We do not have real driver telemetry (speed,
    braking, following distance), so speed is estimated from detection density.
    Stopping distance = reaction (v * 1.5 s) + braking (v² / (2 * a)), a ≈ 6 m/s².
    """
    mph = _assumed_mph(counts)
    scene = classify_scene(counts)
    stype = scene["scene_type"]
    if mph <= 15:
        context = "congested %s" % stype
    elif mph <= 25:
        context = "slow %s traffic" % stype
    elif mph <= 40:
        context = "moderate %s traffic" % stype
    else:
        context = "flowing %s traffic" % stype
    v_ms = mph * 0.44704
    reaction_time = 1.5
    a = 6.0
    stopping_m = v_ms * reaction_time + (v_ms ** 2) / (2 * a)
    following_m = v_ms * 2.5
    car_lengths = max(2, int(round(following_m / 5.0)))
    return "At ~%d mph in %s: give yourself at least %.0f m (like %d car lengths) so you’ve got room to stop if you need it." % (mph, context, stopping_m, car_lengths)


def _alert_explanation(reason, alert_type):
    """Short, human explanation of why this alert fired (cool/humble tone)."""
    if alert_type == "positive":
        if reason == "improvement":
            return "Things got less sketchy compared to a moment ago."
        if reason == "eco_win":
            return "Your drive just got way more efficient. Nice."
        if reason == "streak":
            return "You’ve been keeping it clean for a bit — we noticed."
        return "Something good happened. We’re here for it."
    if reason == "risk":
        return "Stuff got a bit spicier than the last stretch."
    if reason == "scene":
        return "Road changed, risk didn’t back off yet."
    if reason == "co2":
        return "This bit was heavier on the emissions."
    return "Context shifted enough that we wanted to flag it."



def _alert_impact_score(alert_type, reason, risk, risk_delta, co2_delta_abs, counts):
    """
    0-100 impact score for ranking alerts.
    Higher means more important moment to review first.
    """
    vehicles = _f(counts, "car") + _f(counts, "truck") + _f(counts, "bus") + _f(counts, "motorcycle")
    people = float(_norm_counts(counts)["_people"])
    exposure = min(20.0, vehicles * 2.0 + people * 3.0)
    risk_weight = {"low": 8, "medium": 18, "high": 30}.get(risk, 10)

    if alert_type == "positive":
        base = {"improvement": 38, "eco_win": 34, "streak": 32}.get(reason, 30)
        score = base + min(20, co2_delta_abs * 0.2) + min(12, risk_delta * 6)
    else:
        base = {"risk": 52, "scene": 44, "co2": 40}.get(reason, 36)
        score = base + risk_weight + min(20, co2_delta_abs * 0.25) + exposure

    return int(max(0, min(100, round(score))))


def _load_trip_history():
    """Load persistent trip history from disk."""
    try:
        if not TRIP_HISTORY_PATH.exists():
            return []
        data = json.loads(TRIP_HISTORY_PATH.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_trip_history(history):
    """Persist trip history to disk."""
    TRIP_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRIP_HISTORY_PATH.write_text(json.dumps(history, indent=2))


def _build_personalized_targets(current_trip, history):
    """Build simple personalized targets from recent baseline."""
    style = current_trip.get("driving_style", {})
    current = {
        "smoothness": int(style.get("smoothness_score", 0)),
        "anticipation": int(style.get("anticipation_score", 0)),
        "efficiency": int(style.get("efficiency_score", 0)),
    }
    if not history:
        baseline = current.copy()
    else:
        recent = history[-5:]
        def _avg(key):
            vals = [float(h.get("scores", {}).get(key, 0)) for h in recent]
            return int(round(sum(vals) / len(vals))) if vals else 0
        baseline = {
            "smoothness": _avg("smoothness"),
            "anticipation": _avg("anticipation"),
            "efficiency": _avg("efficiency"),
        }

    targets = {}
    for k in ("smoothness", "anticipation", "efficiency"):
        base = baseline.get(k, current[k])
        curr = current[k]
        target = min(95, max(70, base + 10))
        targets[k] = {
            "current": curr,
            "baseline": base,
            "target": target,
            "gap": max(0, target - curr),
        }
    return targets


def _build_savings_simulation(trip_summary):
    """What-if simulator for projected annual savings."""
    fuel = trip_summary.get("fuel_consumption", {})
    style = trip_summary.get("driving_style", {})
    annual_base = float(fuel.get("potential_savings_year_usd", 0))
    smooth = float(style.get("smoothness_score", 0))
    anticipation = float(style.get("anticipation_score", 0))
    efficiency = float(style.get("efficiency_score", 0))

    # Score-dependent what-if: the worse your current score, the more room to improve.
    # A driver at 90 smoothness gets near-zero projection; one at 20 gets a big one.
    smooth_factor = max(0, (90 - smooth) / 100) * 0.5
    anticipation_factor = max(0, (85 - anticipation) / 100) * 0.4
    efficiency_factor = max(0, (90 - efficiency) / 100) * 0.55

    blended_gain = (smooth_factor + anticipation_factor + efficiency_factor) * annual_base

    return {
        "if_smoothness_plus_10": round(annual_base * smooth_factor, 0),
        "if_anticipation_plus_10": round(annual_base * anticipation_factor, 0),
        "if_efficiency_plus_10": round(annual_base * efficiency_factor, 0),
        "if_all_targets_hit": round(blended_gain, 0),
    }


def _update_trip_history_and_trends(trip_summary):
    """
    Store current trip in persistent history and return trend metrics.
    """
    style = trip_summary.get("driving_style", {})
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eco_score": float(trip_summary.get("efficiency_loss_pct", 0)),
        "avg_co2_gkm": float(trip_summary.get("avg_co2_gkm", 0)),
        "distance_km": float(trip_summary.get("distance_est_km", 0)),
        "style": style.get("driving_style", "unknown"),
        "scores": {
            "smoothness": float(style.get("smoothness_score", 0)),
            "anticipation": float(style.get("anticipation_score", 0)),
            "efficiency": float(style.get("efficiency_score", 0)),
        },
    }

    history = _load_trip_history()
    history.append(row)
    history = history[-100:]  # keep latest 100 trips
    _save_trip_history(history)

    recent = history[-7:]
    prev = history[-2] if len(history) >= 2 else None

    def _avg(field):
        vals = [float(h.get(field, 0)) for h in recent]
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    trends = {
        "trip_count": len(history),
        "avg_co2_last7": _avg("avg_co2_gkm"),
        "avg_distance_last7": _avg("distance_km"),
    }
    if prev:
        trends["delta_co2_vs_prev"] = round(row["avg_co2_gkm"] - float(prev.get("avg_co2_gkm", row["avg_co2_gkm"])), 1)
        trends["delta_smoothness_vs_prev"] = round(row["scores"]["smoothness"] - float(prev.get("scores", {}).get("smoothness", row["scores"]["smoothness"])), 1)
    else:
        trends["delta_co2_vs_prev"] = 0.0
        trends["delta_smoothness_vs_prev"] = 0.0
    return trends, history


def build_playback_alerts(results_list, fps, min_gap_sec=10.0):
    """
    Fire alerts for significant scene changes - both warnings AND positive reinforcement:
    WARNINGS:
    - Risk level jumps up (low→high, or low→medium, medium→high)
    - Scene type changes AND risk is at least medium
    - Large CO2 spike (≥50 g/km jump)
    POSITIVE:
    - Risk drops (high→medium, high→low, medium→low) - celebrate improvement!
    - Sustained low-risk driving (30s+ streak)
    - CO2 improvement (drops ≥40 g/km)
    Chunks are 3 seconds wide so single-frame noise is smoothed out.
    min_gap_sec=10 prevents alert fatigue.
    """
    if not results_list or fps <= 0:
        return []

    risk_order = {"low": 0, "medium": 1, "high": 2}
    alerts = []
    prev_risk = "low"
    prev_scene = None
    prev_co2 = 0.0
    last_t = -min_gap_sec - 1
    low_risk_start = None  # Track start of low-risk streak

    # 3-second chunks — wide enough to smooth detection noise
    chunk_frames = max(1, int(fps * 3))
    for start in range(0, len(results_list), chunk_frames):
        end = min(start + chunk_frames, len(results_list))
        chunk_counts = {}
        for i in range(start, end):
            c = summarize_detections([results_list[i]])
            for k, v in c.items():
                chunk_counts[k] = chunk_counts.get(k, 0) + v
        n = end - start
        chunk_counts = {k: round(v / n, 1) for k, v in chunk_counts.items()}

        eco = eco_safety_rules(chunk_counts)
        risk = eco["risk_level"]
        scene_type = eco.get("scene_type", "residential")
        co2 = eco.get("co2_gkm", 120.0)
        t = (start + n // 2) / fps  # midpoint of chunk

        # Track low-risk streaks for positive milestones
        if risk == "low":
            if low_risk_start is None:
                low_risk_start = t
        else:
            low_risk_start = None

        # Decide whether this moment deserves an alert
        reason = None
        alert_type = "warning"  # or "positive"
        
        risk_delta = risk_order.get(risk, 0) - risk_order.get(prev_risk, 0)
        risk_jumped = risk_delta > 0
        risk_improved = risk_delta < 0
        scene_changed = prev_scene is not None and scene_type != prev_scene
        co2_spiked = (co2 - prev_co2) >= 50
        co2_improved = (prev_co2 - co2) >= 40
        co2_delta_abs = abs(co2 - prev_co2)

        # POSITIVE alerts (improvements)
        if risk_improved and prev_risk in ("medium", "high"):
            reason = "improvement"
            alert_type = "positive"
        elif co2_improved and risk == "low":
            reason = "eco_win"
            alert_type = "positive"
        elif low_risk_start is not None and (t - low_risk_start) >= 30 and (t - last_t) >= 20:
            # Milestone: 30+ seconds of low risk
            reason = "streak"
            alert_type = "positive"
        # WARNING alerts
        elif risk_jumped:
            reason = "risk"
        elif scene_changed and risk in ("medium", "high"):
            reason = "scene"
        elif co2_spiked and risk != "low":
            reason = "co2"

        prev_risk = risk
        prev_scene = scene_type
        prev_co2 = co2

        if reason is None:
            continue
        if t - last_t < min_gap_sec and alert_type != "positive":
            continue
        last_t = t

        # Customize message based on alert type (cool, humble, human tone)
        if alert_type == "positive":
            if reason == "improvement":
                tip = "Things chilled out — nice read on the road."
                driver_assessment = "You dialed it back when it mattered. That’s the move."
            elif reason == "eco_win":
                tip = "Your ride just got cleaner. The planet (and your wallet) are into it."
                driver_assessment = "Smooth driving = less gas, less CO₂. You’re doing it right."
            elif reason == "streak":
                duration_sec = int(t - low_risk_start)
                tip = f"Solid {duration_sec}s of calm driving. No notes."
                driver_assessment = "You’re paying attention and keeping it steady. That’s the vibe."
            distance_tip = ""
        else:
            tip = eco["tips"][0] if eco["tips"] else ""
            driver_assessment = eco.get("driver_assessment", "")
            distance_tip = _physics_tip(eco, chunk_counts)

        time_label = "%d:%02d" % (int(t) // 60, int(t) % 60)
        est_mph = _assumed_mph(chunk_counts)
        explanation = _alert_explanation(reason, alert_type)
        impact_score = _alert_impact_score(alert_type, reason, risk, abs(risk_delta), co2_delta_abs, chunk_counts)

        alerts.append({
            "time": round(t, 1),
            "time_label": time_label,
            "risk_level": risk,
            "alert_type": alert_type,  # "warning" or "positive"
            "scene_type": scene_type,
            "co2_gkm": co2,
            "fuel_penalty_pct": eco.get("fuel_penalty_pct", 0),
            "comparison": eco.get("comparison", ""),
            "driver_assessment": driver_assessment,
            "est_mph": est_mph,
            "tip": tip,
            "distance_tip": distance_tip if alert_type == "warning" else "",
            "reason": reason,
            "explanation": explanation,
            "impact_score": impact_score,
        })
    return alerts


def _norm_counts(counts):
    """Normalize so eco logic works for both COCO and BDD100K-trained models."""
    c = dict(counts)
    people = c.get("person", 0) + c.get("pedestrian", 0) + c.get("rider", 0)
    stops = c.get("stop sign", 0) + c.get("traffic sign", 0) + c.get("traffic light", 0)
    c["_people"] = people
    c["_stops"] = stops
    return c


def _f(counts, k):
    return float(counts.get(k, 0) or 0)


def classify_scene(counts):
    """Classify scene as highway/urban/residential/intersection from detections."""
    c = _norm_counts(counts)
    cars = _f(counts, "car") + _f(counts, "truck") + _f(counts, "bus") + _f(counts, "motorcycle") + _f(counts, "motor") + _f(counts, "bike")
    people = float(c["_people"])
    stops = float(c["_stops"])
    lights = _f(counts, "traffic light")

    if (lights >= 1 or _f(counts, "stop sign") >= 1 or _f(counts, "traffic sign") >= 1) and cars >= 1:
        return {"scene_type": "intersection", "description": "Intersection with lights/signs and traffic — expect some stopping."}
    if people >= 2 or (cars >= 3 and people >= 1):
        return {"scene_type": "urban", "description": "Urban mix — traffic and pedestrians. Stop-and-go is normal here."}
    if cars >= 4 and people == 0 and stops == 0:
        return {"scene_type": "highway", "description": "Highway-style — steady cruising, good for efficiency."}
    return {"scene_type": "residential", "description": "Residential or light traffic — pretty chill."}


def estimate_co2(counts, scene):
    """Estimate CO2 emissions in g/km based on scene and driving conditions."""
    c = _norm_counts(counts)
    cars = _f(counts, "car") + _f(counts, "truck") + _f(counts, "bus") + _f(counts, "motorcycle") + _f(counts, "motor") + _f(counts, "bike")
    people = float(c["_people"])
    scene_type = scene["scene_type"]

    baseline = 120.0  # g/km avg passenger vehicle at steady cruise
    penalty_pct = 0

    if scene_type == "urban":
        # Stop-and-go congestion: +60-80% depending on severity
        if cars >= 5 or (people >= 2 and cars >= 2):
            penalty_pct = 75
        else:
            penalty_pct = 60
    elif scene_type == "intersection":
        # Frequent stops: +30%
        penalty_pct = 30
    elif scene_type == "highway":
        # Efficient cruising: -10%
        penalty_pct = -10
    else:
        # Residential: slight penalty for low-speed driving
        penalty_pct = 10

    # Extra idling factor for heavy traffic with pedestrians
    if people >= 1 and cars >= 3:
        penalty_pct += 15

    co2_gkm = round(baseline * (1 + penalty_pct / 100), 1)

    if co2_gkm <= 110:
        co2_label = "Low — efficient cruising conditions"
    elif co2_gkm <= 140:
        co2_label = "Moderate — some efficiency loss from traffic"
    elif co2_gkm <= 180:
        co2_label = "High — stop-and-go driving increases emissions"
    else:
        co2_label = "Very high — congestion causing significant emissions"

    free_flow = baseline * 0.9  # highway baseline
    excess_pct = round((co2_gkm - free_flow) / free_flow * 100)
    if excess_pct > 0:
        comparison = "This scene produces ~%d%% more CO2 than free-flowing traffic." % excess_pct
    else:
        comparison = "This scene is near optimal — close to free-flow efficiency."

    return {
        "co2_gkm": co2_gkm,
        "co2_label": co2_label,
        "fuel_penalty_pct": penalty_pct,
        "comparison": comparison,
    }


def build_trip_summary(results_list, fps, speed_est=None):
    """Build trip-level summary: total CO2, efficiency, green/red segments."""
    if not results_list or fps <= 0:
        return {}

    n_frames = len(results_list)
    duration_sec = n_frames / fps
    chunk_size = max(1, int(fps))  # 1-second windows

    chunks = []
    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        # Average counts for this chunk
        chunk_counts = {}
        for i in range(start, end):
            c = summarize_detections([results_list[i]])
            for k, v in c.items():
                chunk_counts[k] = chunk_counts.get(k, 0) + v
        n_chunk = end - start
        chunk_counts = {k: round(v / n_chunk, 1) for k, v in chunk_counts.items()}

        scene = classify_scene(chunk_counts)
        co2 = estimate_co2(chunk_counts, scene)
        eco = eco_safety_rules(chunk_counts)

        chunks.append({
            "start_frame": start,
            "end_frame": end,
            "scene": scene,
            "co2": co2,
            "risk_level": eco["risk_level"],
            "counts": chunk_counts,
        })

    total_co2_g = 0.0
    co2_values = []
    risk_counts = {"low": 0, "medium": 0, "high": 0}
    worst_co2 = -1
    best_co2 = float("inf")
    worst_idx = 0
    best_idx = 0
    total_distance_km = 0.0

    for idx, ch in enumerate(chunks):
        chunk_dur_sec = (ch["end_frame"] - ch["start_frame"]) / fps
        speed_kmh = _assumed_mph(ch["counts"]) * 1.60934  # mph → km/h
        dist_km = speed_kmh * (chunk_dur_sec / 3600)
        total_distance_km += dist_km
        chunk_co2 = ch["co2"]["co2_gkm"] * dist_km
        total_co2_g += chunk_co2
        co2_values.append(ch["co2"]["co2_gkm"])
        risk_counts[ch["risk_level"]] = risk_counts.get(ch["risk_level"], 0) + 1

        if ch["co2"]["co2_gkm"] > worst_co2:
            worst_co2 = ch["co2"]["co2_gkm"]
            worst_idx = idx
        if ch["co2"]["co2_gkm"] < best_co2:
            best_co2 = ch["co2"]["co2_gkm"]
            best_idx = idx

    avg_co2 = round(sum(co2_values) / len(co2_values), 1) if co2_values else 0
    free_flow_gkm = 108.0  # baseline * 0.9
    efficiency_loss = round((avg_co2 - free_flow_gkm) / free_flow_gkm * 100, 1) if avg_co2 > free_flow_gkm else 0

    total_chunks = len(chunks)
    green_pct = round(risk_counts.get("low", 0) / total_chunks * 100, 1) if total_chunks else 0
    yellow_pct = round(risk_counts.get("medium", 0) / total_chunks * 100, 1) if total_chunks else 0
    red_pct = round(risk_counts.get("high", 0) / total_chunks * 100, 1) if total_chunks else 0

    def _time_range(idx):
        ch = chunks[idx]
        t0 = ch["start_frame"] / fps
        t1 = ch["end_frame"] / fps
        # Use round so sub-1s chunks don't show as "0:00–0:00"; ensure end > start
        s0 = round(t0)
        s1 = max(s0 + 1, round(t1)) if t1 > t0 else s0 + 1
        return "%d:%02d–%d:%02d" % (s0 // 60, s0 % 60, s1 // 60, s1 % 60)

    # Add comprehensive driving analytics
    driving_style = analyze_driving_style(chunks, fps)
    fuel_consumption = calculate_fuel_consumption(avg_co2, total_distance_km, duration_sec)
    co2_phases = analyze_co2_phases(chunks, fps)
    
    trip_data = {
        "total_co2_g": round(total_co2_g, 1),
        "avg_co2_gkm": avg_co2,
        "efficiency_loss_pct": efficiency_loss,
        "green_pct": green_pct,
        "yellow_pct": yellow_pct,
        "red_pct": red_pct,
        "worst_segment": _time_range(worst_idx),
        "best_segment": _time_range(best_idx),
        "distance_est_km": round(total_distance_km, 2),
        "duration_sec": round(duration_sec, 1),
    }
    
    # Generate actionable insights based on all analytics
    insights = generate_actionable_insights(driving_style, fuel_consumption, co2_phases, trip_data)

    # Conditions detection (day/night/overcast)
    conditions = aggregate_conditions(results_list, fps)

    # Idle and hard braking stats from SpeedEstimator (if available)
    idle_stats = speed_est.get_idle_stats(len(results_list)) if speed_est else {}
    brake_stats = speed_est.get_brake_stats() if speed_est else {}

    return {
        **trip_data,
        "driving_style": driving_style,
        "fuel_consumption": fuel_consumption,
        "co2_phases": co2_phases,
        "conditions": conditions,
        "idle_stats": idle_stats,
        "brake_stats": brake_stats,
        "actionable_insights": insights,
    }


def analyze_driving_style(chunks, fps):
    """
    Classify driving style based on behavior patterns over the entire trip.
    Returns: driving_style, smoothness_score, anticipation_score, efficiency_score, characteristics
    """
    if not chunks:
        return {
            "driving_style": "unknown",
            "smoothness_score": 0,
            "anticipation_score": 0,
            "efficiency_score": 0,
            "characteristics": []
        }
    
    # Analyze patterns
    risk_changes = 0
    co2_spikes = 0
    low_risk_duration = 0
    high_co2_duration = 0
    smooth_segments = 0
    
    for i, ch in enumerate(chunks):
        if i > 0:
            prev_risk_order = {"low": 0, "medium": 1, "high": 2}
            curr_risk = prev_risk_order.get(ch["risk_level"], 0)
            prev_risk = prev_risk_order.get(chunks[i-1]["risk_level"], 0)
            if abs(curr_risk - prev_risk) > 0:
                risk_changes += 1
            
            co2_change = abs(ch["co2"]["co2_gkm"] - chunks[i-1]["co2"]["co2_gkm"])
            if co2_change > 50:
                co2_spikes += 1
        
        if ch["risk_level"] == "low":
            low_risk_duration += 1
        if ch["co2"]["co2_gkm"] > 160:
            high_co2_duration += 1
        if ch["risk_level"] == "low" and ch["co2"]["co2_gkm"] < 120:
            smooth_segments += 1
    
    total_chunks = len(chunks)
    
    # Calculate dimension scores (0-100)
    smoothness_score = max(0, 100 - (risk_changes / total_chunks * 100) - (co2_spikes / total_chunks * 100))
    smoothness_score = min(100, round(smoothness_score))
    
    anticipation_score = round((low_risk_duration / total_chunks) * 100)
    
    efficiency_score = max(0, 100 - (high_co2_duration / total_chunks * 120))
    efficiency_score = min(100, round(efficiency_score))
    
    # Classify overall style
    characteristics = []
    if smoothness_score >= 80:
        characteristics.append("Smooth operator")
    elif smoothness_score < 50:
        characteristics.append("Reactive driver")
    
    if anticipation_score >= 75:
        characteristics.append("Anticipates traffic well")
    elif anticipation_score < 40:
        characteristics.append("Responds to immediate threats")
    
    if efficiency_score >= 80:
        characteristics.append("Eco-conscious")
    elif efficiency_score < 50:
        characteristics.append("Efficiency-aware potential")
    
    # Overall classification
    avg_score = (smoothness_score + anticipation_score + efficiency_score) / 3
    if avg_score >= 80:
        driving_style = "Expert Eco-Driver"
    elif avg_score >= 65:
        driving_style = "Efficient Commuter"
    elif avg_score >= 50:
        driving_style = "Average Driver"
    elif avg_score >= 35:
        driving_style = "Reactive Driver"
    else:
        driving_style = "Aggressive Pattern"
    
    return {
        "driving_style": driving_style,
        "smoothness_score": smoothness_score,
        "anticipation_score": anticipation_score,
        "efficiency_score": efficiency_score,
        "characteristics": characteristics,
    }


def calculate_fuel_consumption(avg_co2_gkm, distance_km, duration_sec):
    """
    Convert CO2 emissions to actual fuel consumption and cost.
    Gasoline: ~2.31 kg CO2 per liter
    Returns: liters used, mpg, cost estimates
    """
    if avg_co2_gkm <= 0 or distance_km <= 0:
        return {
            "liters_used": 0,
            "liters_per_100km": 0,
            "mpg_us": 0,
            "cost_usd": 0,
            "potential_savings_year_usd": 0
        }
    
    # Convert CO2 g/km to fuel consumption
    # gasoline: 2310 g CO2 per liter
    liters_per_km = avg_co2_gkm / 2310
    liters_per_100km = liters_per_km * 100
    liters_used = distance_km * liters_per_km
    
    # Convert to US MPG
    mpg_us = 235.214 / liters_per_100km if liters_per_100km > 0 else 0
    
    # Cost estimate (US average ~$3.50/gallon = $0.92/liter)
    cost_per_liter = 0.92
    cost_usd = liters_used * cost_per_liter
    
    # Potential savings if driver improved to optimal (100 g/km)
    # Skip for short clips (< 0.5 km) — extrapolation is meaningless
    optimal_co2 = 100  # g/km
    if avg_co2_gkm > optimal_co2 and distance_km >= 0.5:
        optimal_liters_per_km = optimal_co2 / 2310
        optimal_liters_used = distance_km * optimal_liters_per_km
        trip_savings = (liters_used - optimal_liters_used) * cost_per_liter

        # Extrapolate to annual (assume 12,000 miles = 19,312 km/year)
        annual_km = 19312
        trips_per_year = min(annual_km / distance_km, 600) if distance_km > 0 else 0
        potential_savings_year_usd = trip_savings * trips_per_year
    else:
        potential_savings_year_usd = 0
    
    return {
        "liters_used": round(liters_used, 2),
        "liters_per_100km": round(liters_per_100km, 1),
        "mpg_us": round(mpg_us, 1),
        "cost_usd": round(cost_usd, 2),
        "potential_savings_year_usd": round(potential_savings_year_usd, 2) if potential_savings_year_usd > 0 else 0
    }


def analyze_co2_phases(chunks, fps):
    """
    Break down CO2 emissions by driving phase: acceleration, cruising, braking, idling.
    Uses scene types and CO2 levels as proxies for phases.
    """
    if not chunks:
        return {
            "acceleration_pct": 0,
            "cruising_pct": 0,
            "braking_pct": 0,
            "idling_pct": 0,
            "acceleration_co2": 0,
            "cruising_co2": 0,
            "braking_co2": 0,
            "idling_co2": 0
        }
    
    phase_duration = {"acceleration": 0, "cruising": 0, "braking": 0, "idling": 0}
    phase_co2 = {"acceleration": [], "cruising": [], "braking": [], "idling": []}
    
    for i, ch in enumerate(chunks):
        co2 = ch["co2"]["co2_gkm"]
        scene = ch["scene"].get("scene_type", "residential")
        
        # Classify phase based on scene and CO2
        if scene == "highway" and co2 < 120:
            phase = "cruising"
        elif scene == "urban" and co2 > 160:
            # High CO2 in urban = stop-and-go (braking/idling)
            phase = "idling"
        elif scene == "intersection":
            # Intersections = braking/acceleration mix
            if i > 0 and chunks[i-1]["co2"]["co2_gkm"] > co2:
                phase = "braking"
            else:
                phase = "acceleration"
        elif co2 > 150:
            phase = "acceleration"
        elif co2 < 110:
            phase = "cruising"
        else:
            phase = "braking"
        
        phase_duration[phase] += 1
        phase_co2[phase].append(co2)
    
    total = sum(phase_duration.values())
    
    return {
        "acceleration_pct": round(phase_duration["acceleration"] / total * 100, 1) if total else 0,
        "cruising_pct": round(phase_duration["cruising"] / total * 100, 1) if total else 0,
        "braking_pct": round(phase_duration["braking"] / total * 100, 1) if total else 0,
        "idling_pct": round(phase_duration["idling"] / total * 100, 1) if total else 0,
        "acceleration_co2": round(sum(phase_co2["acceleration"]) / len(phase_co2["acceleration"]), 1) if phase_co2["acceleration"] else 0,
        "cruising_co2": round(sum(phase_co2["cruising"]) / len(phase_co2["cruising"]), 1) if phase_co2["cruising"] else 0,
        "braking_co2": round(sum(phase_co2["braking"]) / len(phase_co2["braking"]), 1) if phase_co2["braking"] else 0,
        "idling_co2": round(sum(phase_co2["idling"]) / len(phase_co2["idling"]), 1) if phase_co2["idling"] else 0,
    }


def generate_actionable_insights(driving_style_data, fuel_data, co2_phases, trip_summary):
    """
    Generate specific, actionable recommendations with projected savings.
    """
    insights = []
    
    # Smoothness improvement
    if driving_style_data["smoothness_score"] < 70:
        potential_improvement = (70 - driving_style_data["smoothness_score"]) * 0.5  # % fuel savings
        annual_savings = fuel_data.get("cost_usd", 0) * (potential_improvement / 100) * 200  # ~200 trips/year
        insights.append({
            "category": "Smoothness",
            "issue": "Frequent acceleration/braking patterns detected",
            "recommendation": "Maintain steady speed, anticipate traffic flow",
            "impact": f"Could save ~{round(annual_savings, 0)}USD/year",
            "difficulty": "Medium"
        })
    
    # Anticipation improvement
    if driving_style_data["anticipation_score"] < 60:
        insights.append({
            "category": "Anticipation",
            "issue": "Reacting to immediate conditions vs. planning ahead",
            "recommendation": "Look 12-15 seconds ahead, coast to red lights",
            "impact": "10-15% fuel savings possible",
            "difficulty": "Easy"
        })
    
    # Idling reduction
    if co2_phases.get("idling_pct", 0) > 25:
        idling_waste = co2_phases["idling_pct"] * fuel_data.get("cost_usd", 0) * 0.3
        insights.append({
            "category": "Idling",
            "issue": f"{co2_phases['idling_pct']}% of trip spent in stop-and-go",
            "recommendation": "Consider alternate routes or departure times",
            "impact": f"Save ~{round(idling_waste * 200, 0)}USD/year",
            "difficulty": "Easy"
        })
    
    # Acceleration optimization
    if co2_phases.get("acceleration_co2", 0) > 180:
        insights.append({
            "category": "Acceleration",
            "issue": "High emissions during acceleration phases",
            "recommendation": "Gentle acceleration to 60mph in ~15 seconds",
            "impact": "8-12% efficiency gain",
            "difficulty": "Easy"
        })
    
    # Overall efficiency
    if fuel_data.get("potential_savings_year_usd", 0) > 100:
        insights.append({
            "category": "Efficiency",
            "issue": "Overall driving pattern has optimization potential",
            "recommendation": "Apply all eco-driving techniques consistently",
            "impact": f"Up to {round(fuel_data['potential_savings_year_usd'], 0)}USD/year savings",
            "difficulty": "Medium"
        })
    
    return insights[:2]  # Keep demo concise


def eco_safety_rules(counts):
    """
    Decision + recommendation module driven by scene and driver-skills framing.
    Returns dict: risk_level, eco_score, co2_impact, tips, skill_focus, driver_assessment.
    """
    c = _norm_counts(counts)
    cars = _f(counts, "car") + _f(counts, "truck") + _f(counts, "bus") + _f(counts, "motorcycle") + _f(counts, "motor") + _f(counts, "bike")
    people = float(c["_people"])
    stops = float(c["_stops"])

    tips = []
    skill_focus = []
    # Base score: efficiency potential of the scene (how much room for eco-driving)
    eco_score = 85
    risk = "low"

    # —— Traffic (smoothness & following distance) ——
    if cars >= 5:
        eco_score -= 28
        risk = "high"
        skill_focus.append("Smoothness")
        skill_focus.append("Following distance")
        tips.append("Heavy traffic: use smooth acceleration and keep 2–3 seconds following distance to avoid constant braking and save fuel.")
    elif cars >= 3:
        eco_score -= 18
        if risk == "low":
            risk = "medium"
        if "Smoothness" not in skill_focus:
            skill_focus.append("Smoothness")
        tips.append("Moderate traffic: avoid rush-and-brake; gentle throttle and early coasting improve efficiency.")
    elif cars >= 1:
        eco_score -= 8
        tips.append("Light traffic: maintain steady speed and anticipate lane changes to avoid unnecessary braking.")

    # —— Traffic controls (anticipation) ——
    if stops >= 1:
        eco_score -= 12
        if risk == "low" and cars >= 1:
            risk = "medium"
        if "Anticipation" not in skill_focus:
            skill_focus.append("Anticipation")
        tips.append("Traffic control ahead: coast early and avoid accelerating into a stop—saves fuel and reduces wear.")

    # —— Pedestrians / vulnerable users (situational awareness) ——
    if people >= 1 and cars >= 1:
        eco_score -= 22
        risk = "high"
        if "Situational awareness" not in skill_focus:
            skill_focus.append("Situational awareness")
        tips.append("Pedestrians or riders nearby: reduce speed early and be ready to stop smoothly (safer and more efficient than hard braking).")
    elif people >= 1:
        eco_score -= 10
        if risk == "low":
            risk = "medium"
        if "Situational awareness" not in skill_focus:
            skill_focus.append("Situational awareness")
        tips.append("Pedestrians in scene: stay alert and moderate speed so you can react without harsh braking.")

    # —— Clear scene ——
    if cars == 0 and people == 0:
        eco_score = min(100, eco_score + 5)
        tips.append("Clear road: maintain steady speed and avoid unnecessary acceleration for best efficiency.")

    eco_score = max(0, min(100, round(eco_score)))

    # CO₂ / efficiency impact label (driver-skill interpretation)
    if eco_score >= 80:
        co2_label = "Low — this stretch is kind to efficient driving"
    elif eco_score >= 60:
        co2_label = "Medium — stay smooth and you’re good"
    elif eco_score >= 40:
        co2_label = "Higher — a little anticipation goes a long way here"
    else:
        co2_label = "High — stop-and-go territory; smooth = less fuel burned"

    # Short driver assessment (one line, human tone)
    if cars >= 5 or (people >= 1 and cars >= 1):
        driver_assessment = "Smooth throttle, leave space, brake early when you can. You’ve got this."
    elif stops >= 1 and cars >= 1:
        driver_assessment = "Coast up to lights and signs instead of gunning it then slamming the brakes. Your wallet will thank you."
    elif people >= 1:
        driver_assessment = "Keep your head on a swivel and speed in check so you can react without panic stops."
    elif cars >= 1:
        driver_assessment = "Steady pace, light foot — efficiency heaven."
    else:
        driver_assessment = "Open road vibes. Just keep it steady."

    if not tips:
        tips.append("Easy on the gas and cruise when you can — saves fuel and cuts emissions. Simple as that.")

    # Scene classification and CO2 estimates
    scene = classify_scene(counts)
    co2 = estimate_co2(counts, scene)

    return {
        "risk_level": risk,
        "eco_score": eco_score,
        "co2_impact": co2["co2_label"],
        "co2_gkm": co2["co2_gkm"],
        "fuel_penalty_pct": co2["fuel_penalty_pct"],
        "comparison": co2["comparison"],
        "scene_type": scene["scene_type"],
        "scene_description": scene["description"],
        "tips": tips[:2],
        "skill_focus": skill_focus[:4],
        "driver_assessment": driver_assessment,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default=str(ROOT / "test.jpg"), help="Image or video path")
    ap.add_argument("--segment-out", type=str, default=None, help="Output dir for segment (parallel mode)")
    ap.add_argument("--output-id", type=str, default=None, help="Unique id for this run (app passes this)")
    args = ap.parse_args()
    source = args.source
    segment_out = Path(args.segment_out) if args.segment_out else None

    # Optional: use all cores by splitting video into N segments and running N processes
    if segment_out is None and str(source).lower().endswith(VIDEO_EXTENSIONS):
        try:
            n = int(os.environ.get("ECOROAD_PARALLEL_VIDEO", "0"))
        except ValueError:
            n = 0
        if n > 1:
            _run_parallel_video(source, n)
            return

    model_path = os.environ.get("ECOROAD_MODEL_PATH", str(ROOT / "yolov8n.pt"))
    model_path = Path(model_path).resolve() if model_path else (ROOT / "yolov8n.pt")
    model = YOLO(str(model_path))
    is_video = str(source).lower().endswith(VIDEO_EXTENSIONS)

    device = _inference_device()
    use_half = device == "cuda"  # FP16 on GPU = faster, less VRAM
    print("Using device:", device, "(FP16)" if use_half else "")

    run_name = ("segment_" + segment_out.name) if segment_out else None
    try:
        conf = float(os.environ.get("ECOROAD_CONF", "0.25"))
    except (TypeError, ValueError):
        conf = 0.25
    predict_kw = dict(save=True, imgsz=640, conf=conf, device=device, half=use_half)
    # Always save under DETECT_DIR so we find the output (otherwise Ultralytics uses runs/detect/predict)
    predict_kw["project"] = str(DETECT_DIR)
    predict_kw["name"] = run_name if run_name else "predict"
    predict_kw["exist_ok"] = True

    video_fps = None
    if is_video and device == "cuda":
        # Batched inference keeps GPU busy (higher utilization, faster overall)
        print("Batched video inference (batch size from ECOROAD_BATCH_SIZE, default 32)")
        results_list, latest, video_fps, speed_est_obj = _run_batched_video(model, source, predict_kw, run_name or "predict_batch", device)
        counts = aggregate_counts_from_frames(results_list)
        frames_processed = len(results_list)
    else:
        is_image = not is_video
        if is_image:
            # Image: run with save=False and save the plotted result ourselves so we never use a stale file
            kw_img = {**predict_kw, "save": False}
            results = model.predict(source=source, **kw_img)
        else:
            results = model.predict(source=source, **predict_kw)
        if is_video:
            results_list = list(results) if hasattr(results, "__iter__") and not isinstance(results, list) else results
            counts = aggregate_counts_from_frames(results_list)
            frames_processed = len(results_list)
        else:
            counts = summarize_detections(results)
            frames_processed = 1
        runs_dir = DETECT_DIR
        if run_name and (runs_dir / run_name).exists():
            latest = runs_dir / run_name
        else:
            latest = runs_dir / "predict"
            if not latest.exists():
                preds = sorted(runs_dir.glob("predict*"), key=lambda p: p.stat().st_mtime)
                if preds:
                    latest = preds[-1]
    source_name = Path(source).name
    out_dir = segment_out if segment_out else OUTPUT_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_id = (args.output_id or os.environ.get("ECOROAD_OUTPUT_ID", "")).strip()
    annotated_base = ("annotated_" + output_id) if output_id and not segment_out else "annotated"

    if is_video:
        annotated_src = latest / source_name
        if not annotated_src.exists():
            vid_files = list(latest.glob("*.mp4")) + list(latest.glob("*.avi")) + list(latest.glob("*.mov"))
            annotated_src = vid_files[0] if vid_files else annotated_src
        web_mp4 = out_dir / (annotated_base + ".mp4")
        if _reencode_to_web_mp4(annotated_src, web_mp4):
            annotated_dst = web_mp4
        else:
            annotated_dst = out_dir / (annotated_base + Path(annotated_src).suffix)
            shutil.copyfile(annotated_src, annotated_dst)
    else:
        # Image: we already ran with save=False and have the result; save the plot directly (avoids using wrong file from predict/)
        res = results[0] if hasattr(results, "__getitem__") else next(iter(results))
        img = res.plot()
        if img is not None:
            ext = Path(source).suffix.lower()
            if ext not in (".jpg", ".jpeg", ".png"):
                ext = ".jpg"
            annotated_dst = out_dir / (annotated_base + ext)
            cv2.imwrite(str(annotated_dst), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        else:
            annotated_src = latest / source_name
            if not annotated_src.exists():
                imgs = list(latest.glob("*.jpg")) + list(latest.glob("*.png")) + list(latest.glob("*.jpeg"))
                annotated_src = imgs[0] if imgs else annotated_src
            annotated_dst = out_dir / (annotated_base + Path(annotated_src).suffix)
            shutil.copyfile(annotated_src, annotated_dst)

    eco = eco_safety_rules(counts)
    media_type = "video" if is_video else "image"

    # Time-synced playback alerts: at which timestamps (seconds) to pause and show a message
    playback_alerts = []
    trip_summary = {}
    duration_sec = 0.0
    # speed_est_obj is only set for batched GPU path; CPU path doesn't have one
    if not is_video or device != "cuda":
        speed_est_obj = None
    if is_video and results_list and frames_processed > 0:
        if video_fps is None:
            cap = cv2.VideoCapture(str(source))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
        else:
            fps = video_fps
        duration_sec = round(frames_processed / fps, 2)
        playback_alerts = build_playback_alerts(results_list, fps)
        trip_summary = build_trip_summary(results_list, fps, speed_est=speed_est_obj)
        if trip_summary:
            # Rank strongest moments first for quick review cards.
            ranked = sorted(playback_alerts, key=lambda a: a.get("impact_score", 0), reverse=True)
            trip_summary["top_alerts"] = ranked[:2]
            trip_summary["alert_mix"] = {
                "warning_count": sum(1 for a in playback_alerts if a.get("alert_type") != "positive"),
                "positive_count": sum(1 for a in playback_alerts if a.get("alert_type") == "positive"),
            }
            trends, history = _update_trip_history_and_trends(trip_summary)
            trip_summary["trends"] = trends
            all_targets = _build_personalized_targets(trip_summary, history[:-1])
            trip_summary["personalized_targets"] = dict(list(all_targets.items())[:2])
            trip_summary["savings_simulation"] = _build_savings_simulation(trip_summary)

    print("\n=== EcoRoad AI Summary ===")
    print("Media:", media_type, f"({frames_processed} frames)" if is_video else "")
    print("Detections (avg per frame):", counts if counts else "None")
    print("Scene:", eco.get("scene_type", ""), "|", eco.get("scene_description", ""))
    print("Risk Level:", eco["risk_level"], "| Eco Score:", eco["eco_score"], "| CO2:", eco["co2_impact"])
    print("CO2: %.1f g/km | Fuel penalty: %d%% | %s" % (eco.get("co2_gkm", 0), eco.get("fuel_penalty_pct", 0), eco.get("comparison", "")))
    if playback_alerts:
        times = [a["time"] for a in playback_alerts]
        show = times if len(times) <= 10 else times[:8]
        suffix = " ..." if len(times) > 10 else ""
        print("Playback alerts:", len(playback_alerts), "at", show, "s" + suffix)
    print("Driver focus:", eco.get("driver_assessment", ""))
    for t in eco["tips"]:
        print("-", t)

    summary = {
        "source": source,
        "media_type": media_type,
        "frames_processed": frames_processed,
        "detections": counts,
        "annotated_media": annotated_dst.name,
        **eco,
    }
    if is_video:
        summary["playback_alerts"] = playback_alerts
        if trip_summary:
            summary["trip_summary"] = trip_summary
        if duration_sec > 0:
            summary["duration_sec"] = duration_sec
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if not segment_out:
        (OUTPUT_DIR / "last_media_type.txt").write_text(media_type)
    print("Saved summary and", annotated_dst.name)


if __name__ == "__main__":
    main()
