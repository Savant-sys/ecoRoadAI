"""EcoRoad AI Flask app — entry point."""
import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    # Default 5050; override with PORT=8080 ./run.sh (macOS often uses 5000/5001)
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=True)
