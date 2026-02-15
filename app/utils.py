"""Helpers for finding and sending annotated output files."""
from pathlib import Path

from flask import send_file

from app.config import OUTPUT_DIR, UPLOADS_DIR, VIDEO_EXT_MIME


def get_original_path(output_id):
    """Return (path, mimetype) for the uploaded source file for this run (uploads/<id>_*), or (None, None)."""
    if output_id is None or not str(output_id).strip().isdigit():
        return None, None
    oid = str(output_id).strip()
    prefix = oid + "_"
    if not UPLOADS_DIR.exists():
        return None, None
    for f in UPLOADS_DIR.iterdir():
        if f.is_file() and f.name.startswith(prefix):
            ext = f.suffix.lower()
            mime = VIDEO_EXT_MIME.get(ext)
            if mime:
                return f.resolve(), mime
            return f.resolve(), "video/mp4"  # fallback
    return None, None


def get_annotated_path(output_id):
    """Return (path, mimetype) for output/annotated_<id>.<ext>; id must be digits only.
    When output_id is given, only return the id-specific file so we never serve another run's video.
    When output_id is None (e.g. legacy link), try generic annotated.<ext>."""
    if output_id is not None and str(output_id).isdigit():
        out = OUTPUT_DIR.resolve()
        oid = str(output_id).strip()
        for ext, mime in VIDEO_EXT_MIME.items():
            p = out / ("annotated_" + oid + ext)
            if p.exists():
                return p.resolve(), mime
        # No fallback when a specific id was requested — avoid showing the wrong run's video
        return None, None
    # No output_id: try generic (e.g. single run or parallel merge)
    for ext, mime in VIDEO_EXT_MIME.items():
        p = OUTPUT_DIR / ("annotated" + ext)
        if p.exists():
            return p, mime
    return None, None


def send_annotated_response(path, mimetype, as_attachment, download_name):
    """Send file. Use conditional=True (Range support) for playback so video seeking works."""
    r = send_file(
        path,
        mimetype=mimetype,
        conditional=not as_attachment,  # Range requests for playback = seekable timeline
        as_attachment=as_attachment,
        download_name=download_name if as_attachment else None,
    )
    if as_attachment:
        r.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        r.headers["Pragma"] = "no-cache"
        r.headers["Expires"] = "0"
    return r
