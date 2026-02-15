from flask import Flask, request, render_template_string, send_file
import subprocess
import json
from pathlib import Path

app = Flask(__name__)

HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>EcoRoad AI — Autopilot-Style Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; background:#0b0f14; color:#e6e6e6; max-width: 1100px; margin: 30px auto; padding: 0 16px; }
    h1 { margin: 0; font-size: 28px; letter-spacing: 0.3px; }
    .sub { color:#9aa4b2; margin-top:6px; }
    .topbar { display:flex; justify-content:space-between; align-items:flex-end; gap: 16px; }
    .card { background:#121826; border:1px solid #1f2a3a; border-radius: 16px; padding: 16px; margin-top: 16px; box-shadow: 0 10px 30px rgba(0,0,0,.25); }
    .row { display:flex; gap: 16px; flex-wrap: wrap; }
    .left { flex: 1 1 620px; }
    .right { flex: 1 1 360px; }
    .pill { display:inline-block; padding:8px 12px; border-radius:999px; background:#0f172a; border:1px solid #273449; margin: 6px 8px 0 0; font-size: 13px; }
    .pill strong { color:#fff; }
    .label { color:#9aa4b2; font-size: 12px; margin-top: 10px; }
    .value { font-size: 20px; margin-top: 4px; }
    img { width:100%; border-radius: 14px; border:1px solid #1f2a3a; }
    input[type=file] { color:#cbd5e1; }
    button { padding:10px 14px; border-radius:12px; border:1px solid #334155; background:#0f172a; color:#e6e6e6; cursor:pointer; }
    button:hover { background:#111c33; }
    pre { background:#0f172a; border:1px solid #273449; border-radius: 12px; padding: 12px; overflow:auto; }
    .grid { display:grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 12px; }
    .mini { background:#0f172a; border:1px solid #273449; border-radius: 14px; padding: 12px; }
    .mini .k { color:#9aa4b2; font-size: 12px; }
    .mini .v { font-size: 22px; margin-top: 4px; }
    .tips li { margin-bottom: 8px; }
  </style>
</head>
<body>
  <div class="topbar">
    <div>
      <h1>EcoRoad AI</h1>
      <div class="sub">Autopilot-style perception + eco-driving insights (SF Hacks demo)</div>
    </div>
  </div>

  <div class="card">
    <form method="POST" enctype="multipart/form-data">
      <div class="row" style="align-items:center;">
        <div style="flex:1 1 420px;">
          <div class="label">Upload road scene image</div>
          <input type="file" name="image" accept="image/*" required>
        </div>
        <div>
          <button type="submit">Analyze Scene</button>
        </div>
      </div>
    </form>
  </div>

  {% if summary %}
  <div class="row">
    <div class="card left">
      <div class="label">Perception Output</div>
      <img src="/annotated" alt="Annotated output">
    </div>

    <div class="card right">
      <div class="label">Autonomy Status</div>

      <div class="grid">
        <div class="mini">
          <div class="k">Risk Level</div>
          <div class="v">{{ summary.risk_level }}</div>
        </div>
        <div class="mini">
          <div class="k">Eco Score</div>
          <div class="v">{{ summary.eco_score }}/100</div>
        </div>
        <div class="mini">
          <div class="k">CO₂ Impact</div>
          <div class="v">{{ summary.co2_impact }}</div>
        </div>
      </div>

      <div style="margin-top:14px;">
        <span class="pill"><strong>Vehicles</strong> {{ (summary.detections.get('car',0) + summary.detections.get('truck',0) + summary.detections.get('bus',0) + summary.detections.get('motorcycle',0)) }}</span>
        <span class="pill"><strong>Pedestrians</strong> {{ summary.detections.get('person',0) }}</span>
        <span class="pill"><strong>Traffic Lights</strong> {{ summary.detections.get('traffic light',0) }}</span>
        <span class="pill"><strong>Stop Signs</strong> {{ summary.detections.get('stop sign',0) }}</span>
      </div>

      <div class="label" style="margin-top:14px;">Perception (raw)</div>
      <pre>{{ summary.detections | tojson(indent=2) }}</pre>

      <div class="label">Eco + Safety Recommendations</div>
      <ul class="tips">
        {% for t in summary.tips %}
          <li>{{ t }}</li>
        {% endfor %}
      </ul>
    </div>
  </div>
  {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    summary = None
    if request.method == "POST":
        f = request.files["image"]
        Path("uploads").mkdir(exist_ok=True)
        img_path = Path("uploads") / f.filename
        f.save(img_path)

        # Run your existing pipeline
        subprocess.run(["python", "ecoroad.py", str(img_path)], check=True)

        summary = json.loads(Path("output/summary.json").read_text())
    return render_template_string(HTML, summary=summary)

@app.route("/annotated")
def annotated():
    return send_file("output/annotated.jpg", mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
