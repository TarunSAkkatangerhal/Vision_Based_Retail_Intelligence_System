"""
Download and prepare the SKU-110K dataset for YOLOv8 training.

SKU-110K is the gold standard for dense retail shelf product detection:
  - 11,762 images of real store shelves
  - 1.73 million bounding box annotations
  - Average 147 products per image
  - Perfect for detecting 50+ items on crowded Indian grocery shelves

Source: https://github.com/eg4000/SKU110K_CVPR19

This script:
  1. Downloads the SKU-110K dataset (~6 GB)
  2. Converts CSV annotations → YOLO format (.txt per image)
  3. Organises into train/val/test splits

Usage:
    python dataset/download_sku110k.py
"""

import csv
import os
import shutil
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────

DATASET_ROOT = Path(__file__).resolve().parent / "SKU-110K"
IMAGES_DIR = DATASET_ROOT / "images"
LABELS_DIR = DATASET_ROOT / "labels"

# Google Drive link for the dataset (official mirror)
# If this stops working, download manually from the GitHub repo.
SKU110K_GDRIVE_ID = "1iq93lCdhaPUN0fWbLieMtzfB1850pKwd"

# Direct download URLs for annotations (CSV)
ANNOTATIONS_URL = {
    "train": "https://storage.googleapis.com/projects_public_data/SKU110K/annotations/annotations_train.csv",
    "val": "https://storage.googleapis.com/projects_public_data/SKU110K/annotations/annotations_val.csv",
    "test": "https://storage.googleapis.com/projects_public_data/SKU110K/annotations/annotations_test.csv",
}

# Direct URLs for images (zip files)
IMAGES_URL = {
    "train": "https://storage.googleapis.com/projects_public_data/SKU110K/SKU110K_fixed/images/train.zip",
    "val": "https://storage.googleapis.com/projects_public_data/SKU110K/SKU110K_fixed/images/val.zip",
    "test": "https://storage.googleapis.com/projects_public_data/SKU110K/SKU110K_fixed/images/test.zip",
}


def _download(url: str, dest: Path, desc: str = "") -> None:
    """Download a file with progress reporting."""
    if dest.exists():
        print(f"  [skip] {desc or dest.name} already exists.")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {desc or url} ...")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / 1024 / 1024
            total_mb = total_size / 1024 / 1024
            print(f"\r    {mb:.0f}/{total_mb:.0f} MB ({pct:.0f}%)", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook=_progress)
    print()


def download_images():
    """Download and extract image zip files."""
    import zipfile

    for split, url in IMAGES_URL.items():
        zip_path = DATASET_ROOT / f"{split}_images.zip"
        img_dir = IMAGES_DIR / split

        if img_dir.exists() and any(img_dir.iterdir()):
            print(f"  [skip] {split} images already extracted.")
            continue

        _download(url, zip_path, desc=f"{split} images")

        print(f"  Extracting {split} images ...")
        img_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                if member.endswith((".jpg", ".JPG", ".png", ".jpeg")):
                    filename = os.path.basename(member)
                    if filename:
                        target = img_dir / filename
                        with zf.open(member) as src, open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst)

        print(f"    Extracted {len(list(img_dir.glob('*')))} images to {img_dir}")

        # Optionally remove zip to save space
        # zip_path.unlink()


def download_annotations():
    """Download CSV annotation files."""
    csv_dir = DATASET_ROOT / "annotations"
    csv_dir.mkdir(parents=True, exist_ok=True)

    for split, url in ANNOTATIONS_URL.items():
        dest = csv_dir / f"annotations_{split}.csv"
        _download(url, dest, desc=f"{split} annotations")


def convert_csv_to_yolo():
    """
    Convert SKU-110K CSV annotations to YOLO format.

    SKU-110K CSV format (per row):
        image_name, x1, y1, x2, y2, class, image_width, image_height

    YOLO format (per line in .txt):
        class_id  cx  cy  w  h   (all normalised 0-1)

    Since SKU-110K is single-class (all "product"), class_id is always 0.
    """
    csv_dir = DATASET_ROOT / "annotations"

    for split in ("train", "val", "test"):
        csv_path = csv_dir / f"annotations_{split}.csv"
        if not csv_path.exists():
            print(f"  [skip] {csv_path} not found.")
            continue

        out_dir = LABELS_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)

        # Group annotations by image
        image_annots: dict[str, list] = defaultdict(list)
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 8:
                    continue
                img_name = row[0].strip()
                try:
                    x1, y1, x2, y2 = float(row[1]), float(row[2]), float(row[3]), float(row[4])
                    img_w, img_h = float(row[6]), float(row[7])
                except (ValueError, IndexError):
                    continue
                if img_w <= 0 or img_h <= 0:
                    continue
                # Convert to YOLO normalised cx, cy, w, h
                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                bw = (x2 - x1) / img_w
                bh = (y2 - y1) / img_h
                # Clamp to [0, 1]
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                bw = max(0.0, min(1.0, bw))
                bh = max(0.0, min(1.0, bh))
                image_annots[img_name].append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        # Write one .txt file per image
        count = 0
        for img_name, lines in image_annots.items():
            stem = Path(img_name).stem
            txt_path = out_dir / f"{stem}.txt"
            with open(txt_path, "w") as f:
                f.write("\n".join(lines) + "\n")
            count += 1

        total_boxes = sum(len(v) for v in image_annots.values())
        print(f"  {split}: {count} label files, {total_boxes} bounding boxes")


def write_data_yaml():
    """Write YOLO data.yaml for the SKU-110K dataset."""
    yaml_path = DATASET_ROOT / "data.yaml"
    content = f"""# SKU-110K Dense Retail Shelf Product Detection
# Source: https://github.com/eg4000/SKU110K_CVPR19

path: {DATASET_ROOT.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  - product
"""
    yaml_path.write_text(content)
    print(f"  Written {yaml_path}")


def main():
    print("=" * 60)
    print("SKU-110K Dataset Setup for YOLO Training")
    print("=" * 60)

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Downloading annotations ...")
    download_annotations()

    print("\n[2/4] Downloading images (this may take a while — ~6 GB) ...")
    download_images()

    print("\n[3/4] Converting CSV annotations → YOLO format ...")
    convert_csv_to_yolo()

    print("\n[4/4] Writing data.yaml ...")
    write_data_yaml()

    print("\n" + "=" * 60)
    print("Done!  Dataset ready at:", DATASET_ROOT)
    print("To train:  python dataset/train_sku110k.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
