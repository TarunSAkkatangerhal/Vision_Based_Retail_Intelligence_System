"""
Train a YOLOv8 nano model on the retail product dataset.

Usage:
    python -m vision.train_model

The trained model is saved to models/inventory_yolo.pt.
"""

import logging
import shutil
from pathlib import Path

from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

DATASET_YAML = "dataset/data.yaml"
BASE_MODEL = "yolov8n.pt"               # pretrained YOLOv8 nano
OUTPUT_DIR = Path("models")
OUTPUT_MODEL = OUTPUT_DIR / "inventory_yolo.pt"

EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 16


def train() -> None:
    """Load dataset, train YOLOv8, and save the best weights."""
    logger.info("Starting training with dataset config: %s", DATASET_YAML)
    logger.info("Base model: %s | Epochs: %d | Image size: %d | Batch: %d",
                BASE_MODEL, EPOCHS, IMAGE_SIZE, BATCH_SIZE)

    # Load pretrained YOLOv8 nano
    model = YOLO(BASE_MODEL)

    # Train
    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project="runs/train",
        name="inventory",
        exist_ok=True,
    )

    # The best weights are saved by ultralytics under runs/train/inventory/weights/best.pt
    best_weights = Path("runs/train/inventory/weights/best.pt")
    if not best_weights.exists():
        logger.error("Training completed but best.pt not found at %s", best_weights)
        return

    # Copy best weights to the expected model path
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, OUTPUT_MODEL)
    logger.info("Trained model saved to: %s", OUTPUT_MODEL)

    # Log key metrics if available
    if results and hasattr(results, "results_dict"):
        metrics = results.results_dict
        logger.info("Training metrics: %s", metrics)


if __name__ == "__main__":
    # Set up logging when run directly
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    train()
