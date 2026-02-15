"""Helpers for finding and sending annotated output files."""
from flask import send_file

from app.config import OUTPUT_DIR, VIDEO_EXT_MIME


def get_annotated_path(output_id):
    """Return (path, mimetype) for output/annotated_<id>.<ext>; id must be digits only.
    If id-specific file is missing (e.g. after app restart or parallel merge wrote annotated.mp4),
    fall back to OUTPUT_DIR / annotated.<ext> so the latest result still loads."""
    if output_id is not None and str(output_id).isdigit():
        out = OUTPUT_DIR
        for ext, mime in list(VIDEO_EXT_MIME.items()) + [(".jpg", "image/jpeg"), (".png", "image/png"), (".jpeg", "image/jpeg")]:
            p = out / ("annotated_" + str(output_id) + ext)
            if p.exists():
                return p, mime
        # Fallback: parallel merge or cleared dir — try generic annotated.<ext>
        for ext, mime in list(VIDEO_EXT_MIME.items()) + [(".jpg", "image/jpeg"), (".png", "image/png"), (".jpeg", "image/jpeg")]:
            p = out / ("annotated" + ext)
            if p.exists():
                return p, mime
    return None, None


def send_annotated_response(path, mimetype, as_attachment, download_name):
    """Send file with no-cache headers so the correct video always loads."""
    r = send_file(
        path,
        mimetype=mimetype,
        conditional=not as_attachment,
        as_attachment=as_attachment,
        download_name=download_name if as_attachment else None,
    )
    r.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    return r
