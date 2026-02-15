"""Flask route handlers."""
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime

from flask import request, render_template, make_response, redirect, url_for

from app.config import (
    ROOT,
    UPLOADS_DIR,
    OUTPUT_DIR,
    RUN_HISTORY_PATH,
    SUMMARIES_DIR,
    TRIP_HISTORY_PATH,
    clear_video_dirs,
    get_sample_video_path,
)
from app.utils import get_annotated_path, send_annotated_response

MAX_HISTORY = 30


def _load_history():
    """Return list of run history entries (newest first)."""
    if not RUN_HISTORY_PATH.exists():
        return []
    try:
        data = json.loads(RUN_HISTORY_PATH.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_history_entry(output_id, filename, summary):
    """Append one entry to run history and persist summary for later view."""
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SUMMARIES_DIR / f"{output_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    entry = {
        "output_id": output_id,
        "filename": filename or "video",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "eco_score": summary.get("eco_score"),
        "risk_level": summary.get("risk_level"),
    }
    history = _load_history()
    history.insert(0, entry)
    RUN_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUN_HISTORY_PATH.write_text(json.dumps(history[:MAX_HISTORY], indent=2))


def register_routes(app):
    """Register all routes on the Flask app."""

    @app.route("/", methods=["GET", "POST"])
    def index():
        summary = None
        output_id = None
        annotated_filename = None
        video_available = True
        history = _load_history()

        sample_available = bool(get_sample_video_path())
        if request.method == "GET" and request.args.get("output_id"):
            oid = request.args.get("output_id", "").strip()
            if oid.isdigit():
                summary_path = SUMMARIES_DIR / f"{oid}.json"
                if summary_path.exists():
                    summary = json.loads(summary_path.read_text())
                    output_id = int(oid)
                    annotated_filename = summary.get("annotated_media")
                    path, _ = get_annotated_path(output_id)
                    video_available = path is not None and path.exists()

        elif request.method == "POST":
            use_sample = request.form.get("use_sample") == "1"
            sample_path = get_sample_video_path() if use_sample else None
            f = None if use_sample else (request.files.get("video") or request.files.get("image") or request.files.get("media"))
            if not use_sample and (not f or f.filename == ""):
                return render_template(
                    "index.html",
                    summary=None,
                    output_id=None,
                    annotated_filename=None,
                    video_available=True,
                    history=history,
                    sample_available=bool(get_sample_video_path()),
                )
            if use_sample and not sample_path:
                return render_template(
                    "index.html",
                    summary=None,
                    output_id=None,
                    annotated_filename=None,
                    video_available=True,
                    history=history,
                    sample_available=sample_available,
                )
            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            output_id = int(time.time() * 1000)
            if use_sample:
                media_path = UPLOADS_DIR / f"{output_id}_sample{sample_path.suffix}"
                shutil.copy2(sample_path, media_path)
                orig_filename = "sample" + sample_path.suffix
            else:
                fn = (f.filename or "").lower()
                if not any(fn.endswith(ext) for ext in (".mp4", ".mov", ".webm", ".avi", ".mkv")):
                    return render_template(
                        "index.html",
                        summary=None,
                        output_id=None,
                        annotated_filename=None,
                        video_available=True,
                        history=history,
                        sample_available=bool(get_sample_video_path()),
                    )
                safe_name = (f.filename or "upload").replace("..", "").replace("/", "_")
                media_path = UPLOADS_DIR / f"{output_id}_{safe_name}"
                f.save(media_path)
                orig_filename = f.filename
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pipeline.ecoroad",
                    str(media_path),
                    "--output-id",
                    str(output_id),
                ],
                check=True,
                cwd=str(ROOT),
            )
            summary = json.loads((OUTPUT_DIR / "summary.json").read_text())
            annotated_filename = summary.get("annotated_media")
            _save_history_entry(output_id, orig_filename, summary)
            history = _load_history()

        cache_bust = int(time.time() * 1000) if summary else None
        resp = make_response(render_template(
            "index.html",
            summary=summary,
            output_id=output_id,
            annotated_filename=annotated_filename,
            video_available=video_available,
            history=history,
            cache_bust=cache_bust,
            sample_available=bool(get_sample_video_path()),
        ))
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        return resp

    @app.route("/clear", methods=["POST"])
    def clear_uploads():
        """Delete all uploads, annotated outputs, and analytics history."""
        clear_video_dirs()
        # Reset all analytics/history so nothing is left
        RUN_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        RUN_HISTORY_PATH.write_text("[]")
        TRIP_HISTORY_PATH.write_text("[]")
        if SUMMARIES_DIR.exists():
            for f in SUMMARIES_DIR.iterdir():
                if f.is_file():
                    try:
                        f.unlink()
                    except OSError:
                        pass
        return redirect(url_for("index"))

    @app.route("/annotated")
    @app.route("/annotated/<output_id>")
    def annotated(output_id=None):
        path, mimetype = get_annotated_path(output_id)
        if path is None or not path.exists():
            return "No output yet", 404
        as_attachment = request.args.get("download") == "1"
        return send_annotated_response(path, mimetype, as_attachment, path.name)

    @app.route("/results/<int:output_id>.json")
    def export_results(output_id):
        """Download full results (summary + trip_summary) as JSON."""
        summary_path = SUMMARIES_DIR / f"{output_id}.json"
        if not summary_path.exists():
            return "Not found", 404
        summary = json.loads(summary_path.read_text())
        resp = make_response(json.dumps(summary, indent=2))
        resp.headers["Content-Type"] = "application/json"
        resp.headers["Content-Disposition"] = f'attachment; filename="ecoroad_results_{output_id}.json"'
        return resp

    @app.route("/.well-known/appspecific/com.chrome.devtools.json")
    def chrome_devtools():
        return "", 204
