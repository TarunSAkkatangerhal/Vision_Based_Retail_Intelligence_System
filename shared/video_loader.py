import logging
from typing import Generator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def get_frame_generator(source: str) -> Generator[np.ndarray, None, None]:
    """
    Yield frames one at a time from an MP4 file or RTSP stream.

    Args:
        source: Path to an MP4 file or an RTSP stream URL.

    Yields:
        numpy.ndarray: A single video frame (BGR format).
    """
    logger.info("Opening video source: %s", source)
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
