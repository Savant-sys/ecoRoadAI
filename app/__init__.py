"""Flask app package."""
from flask import Flask

from app.config import ROOT, clear_video_dirs
from app.routes import register_routes


def create_app():
    """Create and configure the Flask app."""
    clear_video_dirs()  # clean uploads, output, detect, output_parallel on each run
    app = Flask(__name__, template_folder=str(ROOT / "templates"))
    register_routes(app)
    return app
