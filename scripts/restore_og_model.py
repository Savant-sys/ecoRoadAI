"""
Remove old model weights and download the original Ultralytics YOLOv8n (COCO) model.
Run from project root: py scripts/restore_og_model.py
"""
import os
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

def main():
    # Delete .pt files in project root (trained or other copies)
    for pt in ROOT.glob("*.pt"):
        print("Removing:", pt)
        pt.unlink()

    # Delete weights from previous training runs
    runs = ROOT / "runs" / "train"
    if runs.is_dir():
        for pt in runs.rglob("*.pt"):
            print("Removing:", pt)
            pt.unlink()

    # Download official YOLOv8n (COCO); Ultralytics saves to current dir
    print("Downloading official yolov8n.pt (COCO)...")
    os.chdir(ROOT)
    YOLO("yolov8n.pt")
    out = ROOT / "yolov8n.pt"
    print("Saved:", out)
    print("Done. App will use this model.")

if __name__ == "__main__":
    main()
