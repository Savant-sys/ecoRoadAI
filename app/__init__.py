"""Flask app package."""
from flask import Flask

from app.config import ROOT, ensure_video_dirs
from app.routes import register_routes


def create_app():
    """Create and configure the Flask app."""
    ensure_video_dirs()  # create dirs only; clear only when user clicks Clear button
    app = Flask(__name__, template_folder=str(ROOT / "templates"))
    register_routes(app)
    return app
