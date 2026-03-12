"""
Main pipeline — Vision-Based Retail Intelligence System.

Reads video frames, runs inventory detection and customer tracking,
and POSTs the results to the backend API.
Shows a live video window with stickman skeletons and auto-detected shelves.

Usage:
    python main.py                          # uses webcam
    python main.py path/to/video.mp4        # custom video file
    python main.py rtsp://camera/stream     # RTSP stream
    python main.py 0                        # webcam

Flags:
    --no-skeleton   Fall back to bounding-box person detection
    --no-auto-shelf Use only manually calibrated shelf regions
"""

import sys
import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from shared.utils import send_to_api
from shared.config import PROCESS_EVERY_N_FRAMES
from shared.video_loader import get_frame_generator
from vision.inventory import detect_inventory, SHELF_REGIONS as MANUAL_SHELF_REGIONS
from vision.product_detection import detect_products
from vision.product_tracker import check_disappeared
from vision.auto_shelf_detection import update_auto_shelves, get_auto_shelves, is_stable as shelves_stable

logger = logging.getLogger(__name__)

DEFAULT_SOURCE = "0"

# Parse simple flags
_USE_SKELETON = "--no-skeleton" not in sys.argv
_USE_AUTO_SHELF = "--no-auto-shelf" not in sys.argv

# Colours for drawing (BGR)
_COLOR_PRODUCT = (0, 255, 0)    # green for products
_COLOR_PERSON = (255, 0, 0)     # blue for people
_COLOR_TEXT_BG = (0, 0, 0)      # black background for text
_COLOR_SHELF = (0, 165, 255)    # orange for auto-detected shelf regions

# Thread pool for non-blocking API calls
_api_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="api")


def _post_async(endpoint: str, data: dict) -> None:
    """Fire-and-forget API POST so it doesn't block the video loop."""
    _api_pool.submit(send_to_api, endpoint, data)


def _interpolate_boxes(prev: list, curr: list, alpha: float) -> list:
    """Linearly interpolate bounding boxes between two detection frames.

    Matches boxes by index (works when detection count is stable).
    Falls back to current if lengths differ.
    """
    if len(prev) != len(curr):
        return curr
    interpolated = []
    for p, c in zip(prev, curr):
        pb, cb = p["bbox"], c["bbox"]
        new_bbox = [
            pb[i] + alpha * (cb[i] - pb[i])
            for i in range(4)
        ]
        interpolated.append({**c, "bbox": new_bbox})
    return interpolated


def _draw_detections(display_frame: np.ndarray,
                     product_detections: list,
                     person_detections: list,
                     customer_data: dict,
                     shelf_regions: list = None,
                     use_skeleton: bool = False) -> np.ndarray:
    """Draw bounding boxes / skeletons and labels on the frame for display."""

    # ── Draw auto-detected shelf regions (orange dashed rectangles) ──
    if shelf_regions:
        for region in shelf_regions:
            x1 = int(region.get("x_min", 0))
            y1 = int(region.get("y_min", 0))
            x2 = int(region.get("x_max", 0))
            y2 = int(region.get("y_max", 0))
            shelf_name = region.get("shelf_id") or region.get("name", "?")
            # Dashed rectangle effect via small segments
            for i in range(x1, x2, 10):
                cv2.line(display_frame, (i, y1), (min(i + 5, x2), y1), _COLOR_SHELF, 1)
                cv2.line(display_frame, (i, y2), (min(i + 5, x2), y2), _COLOR_SHELF, 1)
            for i in range(y1, y2, 10):
                cv2.line(display_frame, (x1, i), (x1, min(i + 5, y2)), _COLOR_SHELF, 1)
                cv2.line(display_frame, (x2, i), (x2, min(i + 5, y2)), _COLOR_SHELF, 1)
            cv2.putText(display_frame, shelf_name, (x1 + 2, y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, _COLOR_SHELF, 1)

    # ── Draw product detections (green) ──
    for det in product_detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        label = f'{det["product_name"]} {det["confidence"]:.0%}'
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), _COLOR_PRODUCT, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(display_frame, (x1, y1 - th - 6), (x1 + tw, y1), _COLOR_PRODUCT, -1)
        cv2.putText(display_frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # ── Draw person detections (skeleton or bounding box) ──
    if use_skeleton and person_detections:
        from vision.skeleton_tracking import draw_skeleton
        draw_skeleton(display_frame, person_detections)
    else:
        for i, det in enumerate(person_detections):
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            label = f'Person {det.get("confidence", 0):.0%}'
            customers = customer_data.get("customers", [])
            if i < len(customers):
                c = customers[i]
                label = f'{c["customer_id"]} | {c["shelf_id"]} | {c["dwell_time_seconds"]}s'
                if c["interaction"] != "none":
                    label += f' [{c["interaction"]}]'
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), _COLOR_PERSON, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display_frame, (x1, y1 - th - 6), (x1 + tw, y1), _COLOR_PERSON, -1)
            cv2.putText(display_frame, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ── Draw a status bar at the top ──
    n_products = len(product_detections)
    n_customers = len(customer_data.get("customers", []))
    shelf_mode = "auto" if (shelf_regions and any(
        r.get("auto_detected") for r in shelf_regions)) else "manual"
    track_mode = "skeleton" if use_skeleton else "bbox"
    status = (f"Products: {n_products} | Customers: {n_customers} | "
              f"Shelves: {shelf_mode} | Tracking: {track_mode} | 'q' to quit")
    cv2.rectangle(display_frame, (0, 0), (len(status) * 9 + 10, 30), _COLOR_TEXT_BG, -1)
    cv2.putText(display_frame, status, (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return display_frame


def main() -> None:
    # Strip flags from argv so source can still be passed
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    source = args[0] if args else DEFAULT_SOURCE
    logger.info("Starting pipeline — source: %s, process every %d frames, "
                "skeleton=%s, auto_shelf=%s",
                source, PROCESS_EVERY_N_FRAMES, _USE_SKELETON, _USE_AUTO_SHELF)

    total_frames = 0
    processed = 0
    successes = 0
    failures = 0

    # Cache latest detections so we can draw them on every frame
    last_product_detections: list = []
    last_person_detections: list = []
    last_customer_data: dict = {"customers": []}
    active_shelf_regions: list = []

    # Previous detections for smooth interpolation
    prev_product_detections: list = []
    prev_person_detections: list = []

    for frame_idx, frame in enumerate(get_frame_generator(source)):
        total_frames += 1
        h, w = frame.shape[:2]

        # Smooth interpolation between detection frames
        if frame_idx % PROCESS_EVERY_N_FRAMES != 0 and PROCESS_EVERY_N_FRAMES > 1:
            alpha = (frame_idx % PROCESS_EVERY_N_FRAMES) / PROCESS_EVERY_N_FRAMES
            draw_products = _interpolate_boxes(prev_product_detections, last_product_detections, alpha)
            if not _USE_SKELETON:
                draw_persons = _interpolate_boxes(prev_person_detections, last_person_detections, alpha)
            else:
                draw_persons = last_person_detections  # skeleton can't be interpolated like boxes
        else:
            draw_products = last_product_detections
            draw_persons = last_person_detections

        # ── Display every frame (with interpolated detections overlaid) ──
        try:
            display = frame.copy()
            display = _draw_detections(
                display, draw_products, draw_persons, last_customer_data,
                shelf_regions=active_shelf_regions,
                use_skeleton=_USE_SKELETON,
            )
            cv2.imshow("Retail Intelligence System", display)
        except Exception as e:
            logger.error("Display error on frame %d: %s", frame_idx, e)

        # Press 'q' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("User pressed 'q' — stopping.")
            break

        # Only run detection on every Nth frame
        if frame_idx % PROCESS_EVERY_N_FRAMES != 0:
            continue

        processed += 1
        logger.info("Processing frame %d …", frame_idx)

        try:
            # Save previous detections for interpolation
            prev_product_detections = last_product_detections
            prev_person_detections = last_person_detections

            # ── Product detection (single pass for both display and inventory) ──
            last_product_detections = detect_products(frame)

            # ── Check for disappeared products (taken items) ──
            taken_events = check_disappeared(
                last_product_detections, frame,
                shelf_regions=active_shelf_regions,
            )

            # ── Auto shelf detection (cluster products into shelf regions) ──
            if _USE_AUTO_SHELF:
                active_shelf_regions = update_auto_shelves(
                    last_product_detections, frame_width=w, frame_height=h,
                )
                if not active_shelf_regions:
                    # Fall back to manual shelves during warmup
                    active_shelf_regions = _get_manual_shelf_regions()
            else:
                active_shelf_regions = _get_manual_shelf_regions()

            # ── Inventory detection (reuses detections, no duplicate inference) ──
            inventory_data = detect_inventory(frame, detections=last_product_detections)
            _post_async("/inventory", inventory_data)

            # Build shelf item counts for interaction detection
            shelf_counts: dict = {}
            for product in inventory_data.get("products", []):
                sid = product["shelf_id"]
                shelf_counts[sid] = shelf_counts.get(sid, 0) + product["count"]

            # ── Customer tracking ──
            if _USE_SKELETON:
                from vision.skeleton_tracking import detect_customers_skeleton
                last_customer_data, last_person_detections = detect_customers_skeleton(
                    frame,
                    shelf_counts=shelf_counts,
                    shelf_regions=active_shelf_regions,
                )
            else:
                from vision.customer_tracking import detect_customers
                last_customer_data, last_person_detections = detect_customers(
                    frame, shelf_counts=shelf_counts,
                )
            # ── Attach taken-item images to matching customer interactions ──
            if taken_events:
                taken_by_shelf = {e["shelf_id"]: e["image_path"] for e in taken_events}
                for cust in last_customer_data.get("customers", []):
                    if cust["interaction"] == "picked_product" and cust["shelf_id"] in taken_by_shelf:
                        cust["item_image_path"] = taken_by_shelf[cust["shelf_id"]]

            _post_async("/customers", last_customer_data)

            successes += 1

        except Exception as e:
            failures += 1
            logger.error("Frame %d failed: %s", frame_idx, e, exc_info=True)
            # Keep running — don't let one bad frame kill the pipeline

    cv2.destroyAllWindows()

    # ── Summary ──
    print()
    print("=" * 50)
    print("PIPELINE SUMMARY")
    print("=" * 50)
    print(f"  Total frames read : {total_frames}")
    print(f"  Frames processed  : {processed}")
    print(f"  Successes         : {successes}")
    print(f"  Failures          : {failures}")
    print(f"  Skeleton mode     : {_USE_SKELETON}")
    print(f"  Auto-shelf mode   : {_USE_AUTO_SHELF}")
    print("=" * 50)

    logger.info("Pipeline finished. %d processed, %d successes, %d failures.",
                processed, successes, failures)


def _get_manual_shelf_regions() -> list:
    """Convert the manually-loaded SHELF_REGIONS dict to the list format."""
    regions = []
    for shelf_id, (x_min, x_max) in MANUAL_SHELF_REGIONS.items():
        regions.append({
            "shelf_id": shelf_id,
            "name": shelf_id,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": 0,
            "y_max": 9999,
        })
    return regions


if __name__ == "__main__":
    main()
