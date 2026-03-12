"""
Main pipeline — Vision-Based Retail Intelligence System.

Reads video frames, runs inventory detection and customer tracking,
and POSTs the results to the backend API.

Usage:
    python main.py                          # uses dataset/sample_video.mp4
    python main.py path/to/video.mp4        # custom video file
    python main.py rtsp://camera/stream     # RTSP stream
"""

import sys
import logging

from shared.utils import send_to_api
from shared.config import PROCESS_EVERY_N_FRAMES
from shared.video_loader import get_frame_generator
from vision.inventory import detect_inventory
from vision.customer_tracking import detect_customers

logger = logging.getLogger(__name__)

DEFAULT_SOURCE = "dataset/sample_video.mp4"


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE
    logger.info("Starting pipeline — source: %s, process every %d frames",
                source, PROCESS_EVERY_N_FRAMES)

    total_frames = 0
    processed = 0
    successes = 0
    failures = 0

    for frame_idx, frame in enumerate(get_frame_generator(source)):
        total_frames += 1

        # Only process every Nth frame
        if frame_idx % PROCESS_EVERY_N_FRAMES != 0:
            continue

        processed += 1
        logger.info("Processing frame %d …", frame_idx)

        try:
            # ── Inventory detection ──
            inventory_data = detect_inventory(frame)
            inv_resp = send_to_api("/inventory", inventory_data)
            if inv_resp is None:
                logger.warning("Frame %d: failed to POST inventory data.", frame_idx)

            # ── Customer tracking ──
            customer_data = detect_customers(frame)
            cust_resp = send_to_api("/customers", customer_data)
            if cust_resp is None:
                logger.warning("Frame %d: failed to POST customer data.", frame_idx)

            successes += 1

        except Exception as e:
            failures += 1
            logger.error("Frame %d failed: %s", frame_idx, e)

    # ── Summary ──
    print()
    print("=" * 50)
    print("PIPELINE SUMMARY")
    print("=" * 50)
    print(f"  Total frames read : {total_frames}")
    print(f"  Frames processed  : {processed}")
    print(f"  Successes         : {successes}")
    print(f"  Failures          : {failures}")
    print("=" * 50)

    logger.info("Pipeline finished. %d processed, %d successes, %d failures.",
                processed, successes, failures)


if __name__ == "__main__":
    main()
