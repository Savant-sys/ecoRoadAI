# EcoRoad AI — SF Hacks 2026

Flask app + YOLO perception + eco scoring.

- **`app.py`** — Dashboard (video upload, pipeline UI)
- **`ecoroad.py`** — PyTorch (Ultralytics YOLO) perception + eco scoring
- **`yolov8n.pt`** — PyTorch weights (replace with BDD100K fine-tuned `best.pt` if you train)
- **`train_bdd100k.py`** — Fine-tune on BDD100K (use **pre-trained**, not from scratch)

## Run

```bash
cd ecoRoadAI
py -m pip install -r requirements.txt
py app.py
```

Open **http://localhost:5000**. Upload an image or short video (road/dashcam).

## Train on BDD100K (transfer learning)

We use a **pre-trained** model (e.g. YOLOv8 trained on COCO) and **fine-tune** it on BDD100K. That's **transfer learning**: the model already knows generic objects; we only update the weights on driving data. No training from scratch.

**For a strong GPU (e.g. RTX 5090, 24GB+ VRAM):** use a bigger batch and more epochs, and optionally a slightly larger model for better accuracy:

```bash
# Better quality: larger model (s), more epochs, bigger batch
py train_bdd100k.py --model yolov8s.pt --epochs 100 --batch 32 --workers 8
# Or push it (yolov8m, batch 24–32)
py train_bdd100k.py --model yolov8m.pt --epochs 120 --batch 24
```

**Quick run (defaults):** `py train_bdd100k.py --data datasets/bdd100k.yaml`

---

### Do I need the bdd100k_to_YOLO folder?

**Only if you download the official BDD100K** from bdd100k.com (labels come as JSON). The converter turns those into YOLO-format `.txt` files so `train_bdd100k.py` can use them.

- **Pre-converted dataset** (e.g. Kaggle "bdd100k YOLO"): you do **not** need the converter. Download it, set `path` in `datasets/bdd100k.yaml`, and run `train_bdd100k.py`.
- **Raw BDD100K:** put it at `../bdd100k`, run `py bdd_to_yolo_converter.py` from `bdd100k_to_YOLO`, then set `datasets/bdd100k.yaml` to that folder.

You can keep the folder; the app does not use it until you train on BDD100K.

### Next steps (to train on BDD100K)

1. Get BDD100K in YOLO format (pre-converted or convert with bdd100k_to_YOLO).
2. In `datasets/bdd100k.yaml` set `path` to the folder that has `images/train`, `images/val`, `labels/train`, `labels/val`.
3. Run: `py train_bdd100k.py --model yolov8s.pt --epochs 100 --batch 32`.
4. Use the trained model: copy `runs/train/bdd100k/weights/best.pt` to `yolov8n.pt` (or point `ecoroad.py` at it).

---

### What's going on (for learning)

- **Pre-trained model:** YOLOv8 was trained on COCO (many object classes). Its "backbone" learned general visual features; the "head" predicts boxes and classes.
- **Transfer learning:** We load those weights and train further on BDD100K. The model **adapts** to driving scenes (cars, pedestrians, traffic lights, signs) instead of learning from scratch. Fewer epochs and less data are needed.
- **Modifying the model:** You can later try freezing the backbone and only training the head (faster, less VRAM), or changing the number of classes in the head to match your research dataset. Ultralytics supports this via the model YAML and `freeze` in the trainer.

Built on **PyTorch** and Ultralytics YOLO.
