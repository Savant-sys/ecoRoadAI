"""Flask route handlers."""
import json
import subprocess
import sys
import time

from flask import request, render_template, make_response

from app.config import ROOT, UPLOADS_DIR, OUTPUT_DIR
from app.utils import get_annotated_path, send_annotated_response


def register_routes(app):
    """Register all routes on the Flask app."""

    @app.route("/", methods=["GET", "POST"])
    def index():
        summary = None
        output_id = None
        annotated_filename = None
        if request.method == "POST":
            f = request.files.get("video") or request.files.get("image") or request.files.get("media")
            if not f or f.filename == "":
                return render_template(
                    "index.html",
                    summary=None,
                    output_id=None,
                    annotated_filename=None,
                )
            fn = (f.filename or "").lower()
            if not any(fn.endswith(ext) for ext in (".mp4", ".mov", ".webm", ".avi", ".mkv")):
                return render_template(
                    "index.html",
                    summary=None,
                    output_id=None,
                    annotated_filename=None,
                )
            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            output_id = int(time.time() * 1000)
            safe_name = (f.filename or "upload").replace("..", "").replace("/", "_")
            media_path = UPLOADS_DIR / f"{output_id}_{safe_name}"
            f.save(media_path)
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
        resp = make_response(render_template(
            "index.html",
            summary=summary,
            output_id=output_id,
            annotated_filename=annotated_filename,
        ))
        # Prevent caching so second upload always shows the new result
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        return resp

    @app.route("/annotated")
    @app.route("/annotated/<output_id>")
    def annotated(output_id=None):
        path, mimetype = get_annotated_path(output_id)
        if path is None or not path.exists():
            return "No output yet", 404
        as_attachment = request.args.get("download") == "1"
        return send_annotated_response(path, mimetype, as_attachment, path.name)

    @app.route("/.well-known/appspecific/com.chrome.devtools.json")
    def chrome_devtools():
        return "", 204
