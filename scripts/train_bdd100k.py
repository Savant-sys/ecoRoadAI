"""
train_bdd100k_ultralytics.py

Fine-tune YOLOv8 on BDD100K (YOLO format) with good defaults for driving scenes.

Key features:
- Uses BDD100K class order from bdd100k_ultralytics.yaml if present (prevents car/truck index swaps)
- Writes a resolved dataset yaml with absolute path
- Preflight checks before training
- Auto device selection (GPU if available)
- Resume training support
- Better defaults for autonomous-driving style data (bigger imgsz, close_mosaic, etc.)

Expected dataset structure:
datasets/bdd100k/
  images/train/
  images/val/
  labels/train/
  labels/val/
  bdd100k_ultralytics.yaml   (recommended)
"""

import argparse
import os
from pathlib import Path

import yaml
import torch
from ultralytics import YOLO


# If this script lives inside <root>/scripts/, then ROOT = <root>
# If it lives at <root>/, ROOT = <root>
THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent if (THIS_FILE.parent / "datasets").exists() else THIS_FILE.parent.parent

BDD100K_DIR = ROOT / "datasets" / "bdd100k"
DEFAULT_DATA_YAML = ROOT / "datasets" / "bdd100k_ultralytics.yaml"  # fallback if user passes --data


def _on_fit_epoch_end(trainer):
    """Print useful metrics per epoch (best effort)."""
    try:
        epoch = getattr(trainer, "epoch", 0)
        total = getattr(trainer, "epochs", 0)

        m = getattr(trainer, "metrics", None) or {}
        # Ultralytics key names vary by version, so try a few
        mAP50 = m.get("metrics/mAP50(B)", m.get("mAP50", 0)) or 0
        mAP5095 = m.get("metrics/mAP50-95(B)", m.get("mAP50-95", 0)) or 0

        box = m.get("train/box_loss", m.get("box_loss", 0)) or 0
        cls = m.get("train/cls_loss", m.get("cls_loss", 0)) or 0
        dfl = m.get("train/dfl_loss", m.get("dfl_loss", 0)) or 0
        loss = box + cls + dfl

        loader = getattr(trainer, "train_loader", None)
        num_samples = len(loader.dataset) if loader is not None and hasattr(loader, "dataset") else 0
        elapsed = getattr(trainer, "epoch_time", None) or 0
        fps = (num_samples / elapsed) if elapsed and elapsed > 0 else 0

        print(
            f"Epoch [{epoch+1}/{total}] | "
            f"mAP@0.5: {mAP50:.4f} | mAP@0.5:0.95: {mAP5095:.4f} | "
            f"Train Loss: {loss:.4f} | FPS: {fps:.1f}"
        )
    except Exception:
        pass


def _pick_device(user_device: str | None) -> str:
    """Return a Ultralytics-friendly device string."""
    if user_device:
        return user_device
    return "0" if torch.cuda.is_available() else "cpu"


def _default_workers() -> int:
    # Good general default: cap workers so Windows doesn't get unstable
    # Adjust if you know your environment is stable.
    cpu = os.cpu_count() or 8
    return min(24, max(8, cpu // 2))


def _assert_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")


def main():
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8 on BDD100K (transfer learning)")

    # Dataset
    p.add_argument(
        "--data",
        type=str,
        default="",
        help="Path to dataset yaml. If blank, script prefers datasets/bdd100k/bdd100k_ultralytics.yaml.",
    )

    # Model / training
    p.add_argument("--model", type=str, default="yolov8s.pt", help="yolov8n.pt (fast), yolov8s.pt (recommended), yolov8m.pt (better)")
    p.add_argument("--epochs", type=int, default=120, help="Try 120–200 depending on time/compute")
    p.add_argument("--patience", type=int, default=30, help="Early stop if no improvement for N epochs")
    p.add_argument("--imgsz", type=int, default=960, help="640 default is ok; 960/1024 often better for driving scenes")
    p.add_argument("--batch", type=int, default=0, help="0 lets Ultralytics auto-batch; otherwise set e.g. 32/64")
    p.add_argument("--device", type=str, default="", help="GPU id like '0' or '0,1', or 'cpu'. Blank = auto")
    p.add_argument("--workers", type=int, default=0, help="Dataloader workers (0 = auto)")
    p.add_argument("--seed", type=int, default=42)

    # Run management
    p.add_argument("--project", type=str, default=str(ROOT / "runs" / "train"), help="Ultralytics project folder")
    p.add_argument("--name", type=str, default="bdd100k_yolov8", help="Run name")
    p.add_argument("--exist-ok", action="store_true", help="Do not error if run folder exists")
    p.add_argument("--resume", action="store_true", help="Resume from last checkpoint in the run folder if available")

    args = p.parse_args()

    # -------- Preflight dataset checks --------
    _assert_dir(BDD100K_DIR, "BDD100K root folder")
    _assert_dir(BDD100K_DIR / "images" / "train", "Training images folder")
    _assert_dir(BDD100K_DIR / "labels" / "train", "Training labels folder")
    _assert_dir(BDD100K_DIR / "images" / "val", "Validation images folder")
    _assert_dir(BDD100K_DIR / "labels" / "val", "Validation labels folder")

    # Prefer class order from datasets/bdd100k/bdd100k_ultralytics.yaml to avoid index mismatch
    dataset_yaml_preferred = BDD100K_DIR / "bdd100k_ultralytics.yaml"

    data_yaml_path = None
    if dataset_yaml_preferred.is_file():
        data_yaml_path = dataset_yaml_preferred
        print(f"Using dataset class order from: {data_yaml_path}")
    else:
        if args.data:
            data_yaml_path = Path(args.data)
        else:
            data_yaml_path = DEFAULT_DATA_YAML

        if not data_yaml_path.is_file():
            raise FileNotFoundError(
                "No dataset yaml found.\n"
                f"Expected: {dataset_yaml_preferred}\n"
                f"Or pass: --data path/to.yaml\n"
                f"Or place one at: {DEFAULT_DATA_YAML}"
            )
        print(f"Using dataset yaml from: {data_yaml_path}")

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    # Force absolute dataset base path so images/labels resolve correctly on any OS
    data_cfg["path"] = BDD100K_DIR.resolve().as_posix()

    # Write resolved yaml for reproducibility
    resolved_yaml = ROOT / "datasets" / "bdd100k_resolved.yaml"
    with open(resolved_yaml, "w", encoding="utf-8") as f:
        yaml.dump(data_cfg, f, default_flow_style=False, sort_keys=False)

    # -------- Device / workers / batch defaults --------
    device = _pick_device(args.device)
    workers = args.workers if args.workers > 0 else _default_workers()
    batch = args.batch if args.batch and args.batch > 0 else -1  # Ultralytics uses -1 for auto-batch

    # -------- Load model --------
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = (ROOT / model_path).resolve()

    # If the file isn't local, Ultralytics can still download known weights (like yolov8s.pt),
    # so we just pass the string.
    model = YOLO(str(model_path) if model_path.exists() else args.model)

    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)

    names = data_cfg.get("names") or {}
    print("\n=== Training config ===")
    print("Resolved data yaml:", resolved_yaml)
    print("Model:", args.model)
    print("Device:", device)
    print("Img size:", args.imgsz)
    print("Batch:", "auto" if batch == -1 else batch)
    print("Workers:", workers)
    print("Classes:", len(names), "| sample:", [names.get(i, "?") for i in range(min(6, len(names)))])
    print("======================\n")

    # -------- Train --------
    results = model.train(
        data=str(resolved_yaml),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=batch,
        device=device,
        workers=workers,
        seed=args.seed,

        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,

        # Strong general defaults
        pretrained=True,
        amp=True,             # mixed precision on GPU
        cache=False,          # set True if you have fast SSD and enough space
        cos_lr=True,
        lr0=0.002,
        lrf=0.01,

        # Augmentations
        mosaic=1.0,
        mixup=0.1,
        close_mosaic=10,      # important: stop mosaic near end for realism
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # Validation / outputs
        val=True,
        plots=True,

        # Resume support
        resume=args.resume,
    )

    print("\nDone.")
    print(f"Best weights should be at: {Path(args.project) / args.name / 'weights' / 'best.pt'}")
    return results


if __name__ == "__main__":
    main()
