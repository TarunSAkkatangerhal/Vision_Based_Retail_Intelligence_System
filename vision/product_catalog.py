"""
Product Catalog — Visual product identification using CLIP embeddings.

The store owner registers products by providing a name and one or more
reference images.  At runtime, a detected product crop is matched against
the catalog by cosine similarity of CLIP embeddings.

Workflow:
  1. Register:  python -m vision.product_catalog register "Parle-G" ref_images/parle_g.jpg
  2. At runtime: identify_product(crop) → ("Parle-G", 0.87)

No training or fine-tuning needed — CLIP is used as a frozen feature extractor.

Catalog storage:
    product_catalog/
        catalog.json          ← {name, image_path, embedding_file} per entry
        embeddings/           ← .npy files with CLIP embeddings
        reference_images/     ← copies of the original reference images
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────

_CATALOG_ROOT = Path("product_catalog")
_CATALOG_JSON = _CATALOG_ROOT / "catalog.json"
_EMBEDDINGS_DIR = _CATALOG_ROOT / "embeddings"
_REF_IMAGES_DIR = _CATALOG_ROOT / "reference_images"

# ──────────────────────────────────────────────
# CLIP model (lazy loaded)
# ──────────────────────────────────────────────

_clip_model = None
_clip_preprocess = None
_clip_device = None
_catalog_cache: Optional[Dict[str, Any]] = None   # loaded catalog + embeddings


def _load_clip():
    """Load CLIP ViT-B/32 model (once).  Works on CPU and CUDA."""
    global _clip_model, _clip_preprocess, _clip_device

    if _clip_model is not None:
        return

    try:
        import torch
        import clip  # pip install openai-clip
    except ImportError:
        raise ImportError(
            "Product catalog requires 'openai-clip'. "
            "Install with:  pip install openai-clip torch torchvision"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)
    _clip_device = device
    logger.info("CLIP ViT-B/32 loaded on %s.", device)


def _embed_image(image_bgr: np.ndarray) -> np.ndarray:
    """Compute a normalised CLIP embedding for a BGR image crop."""
    import torch
    from PIL import Image

    _load_clip()

    # Convert BGR → RGB → PIL
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    # Preprocess and encode
    input_tensor = _clip_preprocess(pil_image).unsqueeze(0).to(_clip_device)
    with torch.no_grad():
        features = _clip_model.encode_image(input_tensor)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().flatten()


# ──────────────────────────────────────────────
# Catalog management
# ──────────────────────────────────────────────

def _ensure_dirs():
    _CATALOG_ROOT.mkdir(exist_ok=True)
    _EMBEDDINGS_DIR.mkdir(exist_ok=True)
    _REF_IMAGES_DIR.mkdir(exist_ok=True)


def _load_catalog() -> List[Dict[str, Any]]:
    """Load catalog from disk."""
    if not _CATALOG_JSON.exists():
        return []
    with open(_CATALOG_JSON, "r") as f:
        return json.load(f)


def _save_catalog(entries: List[Dict[str, Any]]):
    _ensure_dirs()
    with open(_CATALOG_JSON, "w") as f:
        json.dump(entries, f, indent=2)


def register_product(product_name: str, image_path: str) -> None:
    """
    Register a product with a reference image.

    Multiple images can be registered for the same product name to improve
    matching accuracy (different angles, lighting, etc).

    Args:
        product_name: Human-readable name (e.g. "Parle-G Biscuit 50g")
        image_path:   Path to a reference image of the product.
    """
    _ensure_dirs()
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Reference image not found: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Compute embedding
    embedding = _embed_image(img)

    # Save reference image copy
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in product_name)
    existing = _load_catalog()
    idx = sum(1 for e in existing if e["product_name"] == product_name)
    ref_filename = f"{safe_name}_{idx}{image_path.suffix}"
    ref_dest = _REF_IMAGES_DIR / ref_filename
    shutil.copy2(image_path, ref_dest)

    # Save embedding
    emb_filename = f"{safe_name}_{idx}.npy"
    emb_dest = _EMBEDDINGS_DIR / emb_filename
    np.save(str(emb_dest), embedding)

    # Add catalog entry
    existing.append({
        "product_name": product_name,
        "image_path": str(ref_dest),
        "embedding_file": str(emb_dest),
    })
    _save_catalog(existing)
    _invalidate_cache()

    logger.info("Registered '%s' with image %s.", product_name, image_path.name)


def list_products() -> List[str]:
    """Return list of unique registered product names."""
    catalog = _load_catalog()
    return sorted(set(e["product_name"] for e in catalog))


def remove_product(product_name: str) -> int:
    """Remove all entries for a product name.  Returns number removed."""
    catalog = _load_catalog()
    keep = []
    removed = 0
    for entry in catalog:
        if entry["product_name"] == product_name:
            # Delete files
            for key in ("image_path", "embedding_file"):
                p = Path(entry.get(key, ""))
                if p.exists():
                    p.unlink()
            removed += 1
        else:
            keep.append(entry)
    _save_catalog(keep)
    _invalidate_cache()
    return removed


# ──────────────────────────────────────────────
# Runtime identification
# ──────────────────────────────────────────────

def _get_catalog_embeddings() -> Optional[Tuple[List[str], np.ndarray]]:
    """Load catalog names + stacked embeddings matrix (cached)."""
    global _catalog_cache
    if _catalog_cache is not None:
        return _catalog_cache

    catalog = _load_catalog()
    if not catalog:
        return None

    names = []
    embeddings = []
    for entry in catalog:
        emb_path = Path(entry["embedding_file"])
        if emb_path.exists():
            names.append(entry["product_name"])
            embeddings.append(np.load(str(emb_path)))

    if not names:
        return None

    emb_matrix = np.stack(embeddings)  # (N, 512)
    _catalog_cache = (names, emb_matrix)
    return _catalog_cache


def _invalidate_cache():
    global _catalog_cache
    _catalog_cache = None


def identify_product(crop_bgr: np.ndarray, min_similarity: float = 0.65) -> Tuple[Optional[str], float]:
    """
    Identify a product crop against the registered catalog.

    Args:
        crop_bgr:        BGR image crop of a single detected product.
        min_similarity:  Minimum cosine similarity to accept a match.

    Returns:
        (product_name, similarity) if matched, or (None, 0.0) if no match.
    """
    data = _get_catalog_embeddings()
    if data is None:
        return (None, 0.0)

    names, emb_matrix = data
    query = _embed_image(crop_bgr)

    # Cosine similarity (embeddings are already L2-normed)
    similarities = emb_matrix @ query    # (N,)
    best_idx = int(np.argmax(similarities))
    best_sim = float(similarities[best_idx])

    if best_sim >= min_similarity:
        return (names[best_idx], best_sim)

    return (None, best_sim)


def identify_products_batch(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Identify all detected products in a frame by matching crops to the catalog.

    Modifies each detection dict IN-PLACE by adding:
        - "identified_name": str or None
        - "match_similarity": float

    Also returns the detections list for convenience.
    """
    data = _get_catalog_embeddings()
    if data is None:
        # No catalog registered — set defaults
        for det in detections:
            det["identified_name"] = None
            det["match_similarity"] = 0.0
        return detections

    names, emb_matrix = data
    h, w = frame.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        # Clamp to frame bounds
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))

        if x2 <= x1 or y2 <= y1:
            det["identified_name"] = None
            det["match_similarity"] = 0.0
            continue

        crop = frame[y1:y2, x1:x2]
        name, sim = identify_product(crop)
        det["identified_name"] = name
        det["match_similarity"] = sim

    return detections


# ──────────────────────────────────────────────
# Wrist-to-product matching (for interaction tracking)
# ──────────────────────────────────────────────

def find_nearest_product(
    wrist_xy: Tuple[float, float],
    product_detections: List[Dict[str, Any]],
    max_distance: float = 80.0,
) -> Optional[str]:
    """
    Find the product closest to a hand/wrist position.

    Uses the distance from wrist to the nearest point on each product bbox.
    Returns the identified_name (if catalog match exists) or the generic
    product_name from detection.

    Args:
        wrist_xy:           (x, y) wrist position in pixels.
        product_detections: Product detections with bbox and optionally identified_name.
        max_distance:       Maximum pixel distance to consider a match.

    Returns:
        Product name string, or None if no product is close enough.
    """
    if not product_detections:
        return None

    wx, wy = wrist_xy
    best_name = None
    best_dist = float("inf")

    for det in product_detections:
        x1, y1, x2, y2 = det["bbox"]

        # Distance from point to rectangle (0 if inside)
        dx = max(x1 - wx, 0, wx - x2)
        dy = max(y1 - wy, 0, wy - y2)
        dist = (dx * dx + dy * dy) ** 0.5

        if dist < best_dist:
            best_dist = dist
            # Prefer identified_name (catalog match) over generic detection name
            best_name = det.get("identified_name") or det.get("product_name", "unknown")

    if best_dist <= max_distance:
        return best_name

    return None


# ──────────────────────────────────────────────
# CLI for registering products
# ──────────────────────────────────────────────

def _cli():
    """Command-line interface for catalog management."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Product Catalog — register reference images for identification"
    )
    sub = parser.add_subparsers(dest="command")

    # register
    reg = sub.add_parser("register", help="Register a product with a reference image")
    reg.add_argument("name", help="Product name (e.g. 'Parle-G Biscuit 50g')")
    reg.add_argument("image", help="Path to reference image")

    # register-folder
    regf = sub.add_parser("register-folder",
                          help="Register all images in a folder as one product")
    regf.add_argument("name", help="Product name")
    regf.add_argument("folder", help="Folder containing reference images")

    # list
    sub.add_parser("list", help="List all registered products")

    # remove
    rm = sub.add_parser("remove", help="Remove a product from the catalog")
    rm.add_argument("name", help="Product name to remove")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "register":
        register_product(args.name, args.image)
        print(f"Registered '{args.name}'.")

    elif args.command == "register-folder":
        folder = Path(args.folder)
        count = 0
        for img_path in sorted(folder.glob("*")):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                register_product(args.name, str(img_path))
                count += 1
        print(f"Registered '{args.name}' with {count} reference images.")

    elif args.command == "list":
        products = list_products()
        if not products:
            print("No products registered. Use 'register' to add products.")
        else:
            print(f"{len(products)} products:")
            for name in products:
                print(f"  - {name}")

    elif args.command == "remove":
        n = remove_product(args.name)
        print(f"Removed {n} entries for '{args.name}'.")

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
