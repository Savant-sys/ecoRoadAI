# EcoRoad AI — Setup

Quick setup so you can run the app and analyze a video.

## Requirements

- **Python** 3.10 or newer
- **pip** (usually with Python)
- **GPU** optional — speeds up YOLO inference if you have CUDA

## One-time setup

1. **Clone or open** the `ecoRoadAI` folder.

2. **Create a virtual environment:**
   ```bash
   cd ecoRoadAI
   python -m venv .venv
   ```

3. **Activate and install dependencies:**
   - **Windows (PowerShell):**  
     `.venv\Scripts\activate`  
     `pip install -r requirements.txt`
   - **Mac / Linux:**  
     `source .venv/bin/activate` (or `. .venv/bin/activate`)  
     `pip install -r requirements.txt`

4. **Optional:** Put a short video at `samples/sample.mp4` to use **Try with sample video** without uploading.

## Run the app

- **Windows:**  
  `.venv\Scripts\activate` then `python app.py`
- **Mac / Linux:**  
  `./run.sh`

Open **http://localhost:5050**. Upload a video (or use the sample) and wait for analysis.

## Optional environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ECOROAD_MODEL_PATH` | `yolov8n.pt` | Path to YOLO weights file |
| `ECOROAD_CONF` | `0.25` | Detection confidence (0–1) |
| `ECOROAD_DEVICE` | auto | `cuda` or `cpu` |
| `PORT` | `5050` | Server port |

## Troubleshooting

- **Port in use:** Set `PORT=8080` (or another free port) before running.
- **Slow analysis:** Use a GPU and set `ECOROAD_DEVICE=cuda` if you have CUDA.
- **Model not found:** Ensure `yolov8n.pt` is in the project root or set `ECOROAD_MODEL_PATH` to your weights path.
