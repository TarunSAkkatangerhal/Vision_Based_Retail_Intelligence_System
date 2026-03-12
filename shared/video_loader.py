import logging
import sys
from typing import Generator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def get_frame_generator(source: str) -> Generator[np.ndarray, None, None]:
    """
    Yield frames one at a time from an MP4 file, RTSP stream, or webcam.

    Args:
        source: Path to an MP4 file, an RTSP stream URL, or a digit for webcam index.

    Yields:
        numpy.ndarray: A single video frame (BGR format).
    """
    logger.info("Opening video source: %s", source)
    # If source is a digit string like "0", treat it as a webcam index
    if source.isdigit():
        cam_index = int(source)
        # Use DirectShow on Windows for better webcam compatibility
        if sys.platform == "win32":
            cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(cam_index)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error("Failed to open video source: %s", source)
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        yield frame

    cap.release()
    logger.info("Video source ended. Total frames read: %d", frame_count)
