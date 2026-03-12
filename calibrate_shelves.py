#!/usr/bin/env python3
"""
Shelf Calibration Tool
======================
Interactive OpenCV tool to define shelf regions on a camera frame.

Usage:
    python calibrate_shelves.py                     # uses webcam (cam 0)
    python calibrate_shelves.py --source video.mp4  # uses a video file
    python calibrate_shelves.py --source image.jpg  # uses a static image

Controls:
    - Click & drag to draw a box around a shelf
    - After drawing, type the shelf name (e.g. "shelf_1") in the terminal
    - Optionally type expected product names (comma-separated)
    - Press 'u' to undo the last shelf
    - Press 's' to save and exit
    - Press 'q' to quit without saving
    - Press 'r' to reset all shelves and start over
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "shelf_config.json")

# ── Drawing state ──
drawing = False
start_x, start_y = -1, -1
end_x, end_y = -1, -1
shelves = []           # list of dicts: {name, x_min, y_min, x_max, y_max, expected_products}
current_frame = None   # the base frame (clean copy)
display_frame = None   # drawn-on copy shown to user

# ── Colours ──
COLORS = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 0, 255),    # red
    (255, 255, 0),  # cyan
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (128, 255, 0),  # lime
    (0, 128, 255),  # orange
]


def get_color(idx: int):
    return COLORS[idx % len(COLORS)]


def draw_shelves(frame):
    """Draw all saved shelf rectangles + labels on the frame."""
    overlay = frame.copy()
    for i, shelf in enumerate(shelves):
        color = get_color(i)
        x1, y1, x2, y2 = shelf["x_min"], shelf["y_min"], shelf["x_max"], shelf["y_max"]

        # Semi-transparent fill
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        overlay = frame.copy()

        # Solid border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label
        label = shelf["name"]
        if shelf.get("expected_products"):
            label += f" ({', '.join(shelf['expected_products'][:3])})"
            if len(shelf["expected_products"]) > 3:
                label += "..."

        # Background for text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return frame


def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, end_x, end_y, display_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
        end_x, end_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_x, end_y = x, y
            # Redraw with current rectangle
            display_frame = current_frame.copy()
            display_frame = draw_shelves(display_frame)
            cv2.rectangle(display_frame, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y


def save_config():
    """Save shelf configuration to JSON."""
    config = {
        "shelves": [],
        "frame_width": current_frame.shape[1] if current_frame is not None else 640,
        "frame_height": current_frame.shape[0] if current_frame is not None else 480,
    }
    for shelf in shelves:
        config["shelves"].append({
            "name": shelf["name"],
            "x_min": shelf["x_min"],
            "y_min": shelf["y_min"],
            "x_max": shelf["x_max"],
            "y_max": shelf["y_max"],
            "expected_products": shelf.get("expected_products", []),
        })

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  Saved {len(shelves)} shelf region(s) to:")
    print(f"  {CONFIG_PATH}")
    print(f"{'='*50}")
    for s in shelves:
        prods = ", ".join(s.get("expected_products", [])) or "(any)"
        print(f"  {s['name']:15s}  ({s['x_min']},{s['y_min']})-({s['x_max']},{s['y_max']})  products: {prods}")
    print()


def load_existing_config():
    """Load existing shelf config if it exists."""
    global shelves
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
        shelves = config.get("shelves", [])
        print(f"Loaded {len(shelves)} existing shelf region(s) from {CONFIG_PATH}")
        return True
    return False


def print_instructions():
    print("\n" + "=" * 55)
    print("  SHELF CALIBRATION TOOL")
    print("=" * 55)
    print("  1. Click & drag to draw a box around each shelf")
    print("  2. Type a name for the shelf in the terminal")
    print("  3. Type expected products (comma-separated) or Enter to skip")
    print()
    print("  Keyboard shortcuts (in the window):")
    print("    s  = Save and exit")
    print("    u  = Undo last shelf")
    print("    r  = Reset all shelves")
    print("    q  = Quit without saving")
    print("=" * 55 + "\n")


def grab_frame(source):
    """Get a single frame from the source (webcam, video, or image)."""
    # Check if source is an image file
    if isinstance(source, str) and source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        frame = cv2.imread(source)
        if frame is None:
            print(f"ERROR: Could not read image: {source}")
            sys.exit(1)
        return frame

    # Otherwise treat as video/webcam
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Could not open video source: {source}")
        sys.exit(1)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("ERROR: Could not read a frame from the source.")
        sys.exit(1)

    return frame


def main():
    global current_frame, display_frame, shelves, drawing, start_x, start_y, end_x, end_y

    parser = argparse.ArgumentParser(description="Shelf Calibration Tool")
    parser.add_argument("--source", default=0,
                        help="Video/image source. Use 0 for webcam, or path to video/image file.")
    args = parser.parse_args()

    # Try to parse as int (webcam index)
    try:
        source = int(args.source)
    except (ValueError, TypeError):
        source = args.source

    print_instructions()

    # Load existing config
    load_existing_config()

    # Grab a frame
    print(f"Grabbing frame from: {source} ...")
    current_frame = grab_frame(source)
    h, w = current_frame.shape[:2]
    print(f"Frame size: {w}x{h}")

    # Set up window
    win_name = "Shelf Calibration - Draw boxes around shelves"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, min(w, 1280), min(h, 720))
    cv2.setMouseCallback(win_name, mouse_callback)

    display_frame = current_frame.copy()
    display_frame = draw_shelves(display_frame)

    waiting_for_input = False

    while True:
        # Add instructions overlay
        show = display_frame.copy()
        cv2.putText(show, "Draw box around shelf | s:Save | u:Undo | r:Reset | q:Quit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(show, f"Shelves defined: {len(shelves)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(win_name, show)
        key = cv2.waitKey(30) & 0xFF

        # Check if a rectangle was just completed
        if not drawing and start_x >= 0 and end_x >= 0 and not waiting_for_input:
            x1, y1 = min(start_x, end_x), min(start_y, end_y)
            x2, y2 = max(start_x, end_x), max(start_y, end_y)

            # Only process if the box is big enough (not an accidental click)
            if abs(x2 - x1) > 20 and abs(y2 - y1) > 20:
                waiting_for_input = True

                # Ask for shelf name in terminal
                default_name = f"shelf_{len(shelves) + 1}"
                name = input(f"\n  Enter shelf name [{default_name}]: ").strip()
                if not name:
                    name = default_name

                # Ask for expected products
                products_str = input(f"  Expected products (comma-separated, or Enter to skip): ").strip()
                expected_products = [p.strip() for p in products_str.split(",") if p.strip()] if products_str else []

                shelf = {
                    "name": name,
                    "x_min": x1,
                    "y_min": y1,
                    "x_max": x2,
                    "y_max": y2,
                    "expected_products": expected_products,
                }
                shelves.append(shelf)
                print(f"  Added: {name} at ({x1},{y1})-({x2},{y2})")
                if expected_products:
                    print(f"  Expected products: {', '.join(expected_products)}")

                # Reset drawing state
                display_frame = current_frame.copy()
                display_frame = draw_shelves(display_frame)
                waiting_for_input = False

            # Reset coordinates
            start_x, start_y = -1, -1
            end_x, end_y = -1, -1

        # Key handlers
        if key == ord('s'):
            if shelves:
                save_config()
                break
            else:
                print("  No shelves defined yet! Draw at least one shelf box.")

        elif key == ord('u'):
            if shelves:
                removed = shelves.pop()
                print(f"  Undid: {removed['name']}")
                display_frame = current_frame.copy()
                display_frame = draw_shelves(display_frame)
            else:
                print("  Nothing to undo.")

        elif key == ord('r'):
            shelves.clear()
            print("  Reset all shelves.")
            display_frame = current_frame.copy()

        elif key == ord('q'):
            print("  Quit without saving.")
            break

        # Check if window was closed
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
