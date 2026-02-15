#!/usr/bin/env bash
# Run the Flask app on Mac/Linux (use venv Python so dependencies are found).
cd "$(dirname "$0")"
.venv/bin/python app.py
