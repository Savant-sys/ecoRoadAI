"""Helpers for finding and sending annotated output files."""
from flask import send_file

from app.config import OUTPUT_DIR, VIDEO_EXT_MIME


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
    """Send file with no-cache headers so the correct video always loads."""
    r = send_file(
        path,
        mimetype=mimetype,
        conditional=False,  # disable 304 so browser always gets fresh content for this run
        as_attachment=as_attachment,
        download_name=download_name if as_attachment else None,
    )
    r.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r
