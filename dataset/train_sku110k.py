"""
Train a YOLOv8x model on the SKU-110K dense shelf dataset.

Optimised for **RTX A5000** (24 GB VRAM):
  - YOLOv8x base (largest, most accurate YOLOv8 variant)
  - 1280px input (detects small items on packed shelves)
  - Batch size 4 at 1280px fits comfortably in 24 GB
  - Heavy mosaic + mixup augmentation for dense scene robustness
  - Cosine LR schedule for stable convergence

Prerequisites:
    python dataset/download_sku110k.py   # download + convert dataset first

Usage:
    python dataset/train_sku110k.py                # full training
    python dataset/train_sku110k.py --resume       # resume interrupted run
    python dataset/train_sku110k.py --epochs 50    # override epochs

The best weights are copied to models/inventory_yolo.pt and the
pipeline will automatically pick them up.
"""

import argparse
import logging
import shutil
from pathlib import Path

from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────

DATASET_YAML = Path("dataset/SKU-110K/data.yaml")
BASE_MODEL = "yolov8x.pt"                        # largest YOLOv8 — best accuracy
OUTPUT_DIR = Path("models")
OUTPUT_MODEL = OUTPUT_DIR / "inventory_yolo.pt"

# ──────────────────────────────────────────────
# Training hyper-parameters (tuned for RTX A5000 24 GB)
# ──────────────────────────────────────────────

DEFAULTS = dict(
    epochs=100,
    imgsz=1280,            # large input → small products become visible
    batch=4,               # 4 × 1280px fits 24 GB VRAM comfortably
    patience=15,           # early stopping if val mAP stalls 15 epochs
    optimizer="AdamW",
    lr0=1e-3,
    lrf=0.01,              # final LR = lr0 * lrf (cosine decay)
    cos_lr=True,           # cosine annealing schedule
    warmup_epochs=5,
    warmup_bias_lr=0.01,
    weight_decay=0.0005,
    # Augmentation — aggressive for dense scenes
    mosaic=1.0,            # always mosaic (great for dense detection)
    mixup=0.15,            # light mixup
    degrees=5.0,           # small rotation
    translate=0.1,
    scale=0.5,             # random scale ±50%
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.4,
    hsv_v=0.4,
    # Architecture
    close_mosaic=10,       # disable mosaic for last 10 epochs (stabilise)
    workers=8,
    amp=True,              # mixed precision — faster + saves VRAM
    project="runs/train",
    name="sku110k_yolov8x",
    exist_ok=True,
    save=True,
    save_period=10,        # checkpoint every 10 epochs
    plots=True,
    val=True,
)


def train(resume: bool = False, **overrides) -> None:
    """Train YOLOv8x on SKU-110K."""
    if not DATASET_YAML.exists():
        print("ERROR: Dataset not found. Run this first:")
        print("    python dataset/download_sku110k.py")
        return

    config = {**DEFAULTS, **overrides}
    epochs = config.pop("epochs")

    print("=" * 60)
    print(f"Training YOLOv8x on SKU-110K  |  {epochs} epochs @ {config['imgsz']}px")
    print(f"Batch {config['batch']}  |  AMP {'ON' if config['amp'] else 'OFF'}")
    print("=" * 60)

    if resume:
        # Resume from last checkpoint
        last_ckpt = Path("runs/train/sku110k_yolov8x/weights/last.pt")
        if not last_ckpt.exists():
            print(f"No checkpoint found at {last_ckpt}, starting fresh.")
            model = YOLO(BASE_MODEL)
        else:
            print(f"Resuming from {last_ckpt}")
            model = YOLO(str(last_ckpt))
    else:
        model = YOLO(BASE_MODEL)

    results = model.train(data=str(DATASET_YAML), epochs=epochs, **config)

    # ── Copy best weights to models/ ──
    best_weights = Path("runs/train/sku110k_yolov8x/weights/best.pt")
    if not best_weights.exists():
        # ultralytics sometimes nests under runs/detect/
        alt = Path("runs/detect/runs/train/sku110k_yolov8x/weights/best.pt")
        if alt.exists():
            best_weights = alt

    if best_weights.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, OUTPUT_MODEL)
        print(f"\n✓ Best weights saved to: {OUTPUT_MODEL}")
    else:
        print(f"\n✗ best.pt not found — check runs/train/sku110k_yolov8x/")

    # Print final metrics
    if results and hasattr(results, "results_dict"):
        m = results.results_dict
        print(f"\nFinal metrics:")
        for k, v in m.items():
            if "map" in k.lower() or "precision" in k.lower() or "recall" in k.lower():
                print(f"  {k}: {v:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8x on SKU-110K")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    parser.add_argument("--imgsz", type=int, default=DEFAULTS["imgsz"])
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    train(
        resume=args.resume,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
