"""App config and constants."""
import shutil
from pathlib import Path

# Project root (ecoRoadAI folder)
ROOT = Path(__file__).resolve().parent.parent

# Video/media paths under videos/
VIDEOS_DIR = ROOT / "videos"
UPLOADS_DIR = VIDEOS_DIR / "uploads"      # uploaded source files
OUTPUT_DIR = VIDEOS_DIR / "output"         # annotated results, summary.json
DETECT_DIR = VIDEOS_DIR / "detect"        # YOLO predict output (predict_batch, etc.)
OUTPUT_PARALLEL_DIR = VIDEOS_DIR / "output_parallel"  # segment outputs when using ECOROAD_PARALLEL_VIDEO
ANALYTICS_DIR = VIDEOS_DIR / "analytics"  # persistent cross-trip analytics (not cleared on startup)
TRIP_HISTORY_PATH = ANALYTICS_DIR / "trip_history.json"
RUN_HISTORY_PATH = ANALYTICS_DIR / "run_history.json"  # list of past analyzed videos
SUMMARIES_DIR = ANALYTICS_DIR / "summaries"  # persisted summary per output_id for history view

# Directories that can be cleared via the "Clear uploads & results" button
VIDEO_CLEANUP_DIRS = (UPLOADS_DIR, OUTPUT_DIR, DETECT_DIR, OUTPUT_PARALLEL_DIR)


def ensure_video_dirs():
    """Create upload/output dirs if missing. Does not delete anything."""
    for d in (*VIDEO_CLEANUP_DIRS, ANALYTICS_DIR, SUMMARIES_DIR.parent):
        d.mkdir(parents=True, exist_ok=True)


def clear_video_dirs():
    """Remove all contents of uploads, output, detect, output_parallel. Call only when user clicks Clear."""
    for d in VIDEO_CLEANUP_DIRS:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            continue
        for child in d.iterdir():
            if child.name == ".gitkeep":
                continue
            if child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

# MIME types for annotated video (ecoroad keeps source extension, e.g. .mov)
VIDEO_EXT_MIME = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".webm": "video/webm",
}
