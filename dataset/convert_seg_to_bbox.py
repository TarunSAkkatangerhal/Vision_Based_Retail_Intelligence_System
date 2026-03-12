"""
Convert YOLOv8 segmentation labels (polygon format) to detection labels (bbox format).

Polygon format:  class_id x1 y1 x2 y2 x3 y3 ...
Bbox format:     class_id x_center y_center width height

Overwrites label files in-place.
"""

import glob
import os


def polygon_to_bbox(coords: list[float]) -> tuple[float, float, float, float]:
    """Convert a list of x,y polygon points to x_center, y_center, w, h."""
    xs = coords[0::2]  # even indices = x
    ys = coords[1::2]  # odd indices = y
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    return x_center, y_center, width, height


def convert_file(filepath: str) -> None:
    """Convert a single label file from polygon to bbox format."""
    with open(filepath, "r") as f:
        raw = f.read().strip()

    if not raw:
        return  # empty file, skip

    # Some polygon labels may wrap across multiple lines for a single object.
    # Roboflow sometimes puts everything on one logical line per object,
    # but long polygons may soft-wrap.  Re-join and split by class id.
    # Strategy: split by lines, each line starting with an integer is a new object.
    lines = raw.split("\n")

    # Reassemble: each object starts with an integer class id
    objects = []
    current = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        # Check if this line starts a new object (first token is an integer)
        try:
            int(parts[0])
            # If current has data and this looks like a new class id, flush
            if current:
                objects.append(current)
            current = parts
        except ValueError:
            # Continuation of previous object
            current.extend(parts)
    if current:
        objects.append(current)

    bbox_lines = []
    for obj in objects:
        class_id = obj[0]
        coords = [float(v) for v in obj[1:]]
        if len(coords) < 4:
            continue  # skip malformed
        if len(coords) == 4:
            # Already in bbox format
            bbox_lines.append(f"{class_id} {coords[0]} {coords[1]} {coords[2]} {coords[3]}")
        else:
            # Polygon → bbox
            xc, yc, w, h = polygon_to_bbox(coords)
            bbox_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    with open(filepath, "w") as f:
        f.write("\n".join(bbox_lines) + "\n")


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    label_dirs = [
        os.path.join(base, "labels", "train"),
        os.path.join(base, "labels", "val"),
    ]
    total = 0
    for label_dir in label_dirs:
        files = glob.glob(os.path.join(label_dir, "*.txt"))
        print(f"Converting {len(files)} files in {label_dir} ...")
        for fp in files:
            convert_file(fp)
            total += 1
    print(f"Done. Converted {total} label files to bbox format.")


if __name__ == "__main__":
    main()
