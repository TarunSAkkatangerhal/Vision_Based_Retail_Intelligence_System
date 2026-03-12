"""
Skeleton-based customer tracking using MediaPipe PoseLandmarker (Tasks API).

Replaces bounding-box-only person detection with full-body pose
estimation for:
  - Stickman visualisation (cleaner than bounding boxes)
  - Hand keypoint tracking (better pickup detection — a hand entering
    the shelf region is a much stronger signal than "person bbox overlaps shelf")
  - Body-centre tracking for dwell time

Architecture:
  1. YOLOv8 detects person bounding boxes (fast, multi-person).
  2. Each person crop is fed to MediaPipe PoseLandmarker for skeleton.
  3. Body centres are fed to DeepSort for persistent track IDs.
"""

import logging
import os
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from shared.config import (
    CAMERA_ID,
    MOCK_MODE,
    PERSON_CONFIDENCE_THRESHOLD,
)
from shared.schemas import CustomerBehaviorData, CustomerItem
from shared.utils import get_current_timestamp
from vision.dwell_time import calculate_dwell_time
from vision.interaction import detect_interaction_on_exit

logger = logging.getLogger(__name__)

# ── MediaPipe pose landmarks of interest ──
# Full list: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
_LM_NOSE = 0
_LM_LEFT_SHOULDER = 11
_LM_RIGHT_SHOULDER = 12
_LM_LEFT_ELBOW = 13
_LM_RIGHT_ELBOW = 14
_LM_LEFT_WRIST = 15
_LM_RIGHT_WRIST = 16
_LM_LEFT_HIP = 23
_LM_RIGHT_HIP = 24
_LM_LEFT_KNEE = 25
_LM_RIGHT_KNEE = 26
_LM_LEFT_ANKLE = 27
_LM_RIGHT_ANKLE = 28

# Pairs of landmark indices to draw the stickman skeleton
_SKELETON_CONNECTIONS = [
    (_LM_LEFT_SHOULDER, _LM_RIGHT_SHOULDER),   # shoulders
    (_LM_LEFT_SHOULDER, _LM_LEFT_ELBOW),
    (_LM_LEFT_ELBOW, _LM_LEFT_WRIST),
    (_LM_RIGHT_SHOULDER, _LM_RIGHT_ELBOW),
    (_LM_RIGHT_ELBOW, _LM_RIGHT_WRIST),
    (_LM_LEFT_SHOULDER, _LM_LEFT_HIP),
    (_LM_RIGHT_SHOULDER, _LM_RIGHT_HIP),
    (_LM_LEFT_HIP, _LM_RIGHT_HIP),             # hips
    (_LM_LEFT_HIP, _LM_LEFT_KNEE),
    (_LM_LEFT_KNEE, _LM_LEFT_ANKLE),
    (_LM_RIGHT_HIP, _LM_RIGHT_KNEE),
    (_LM_RIGHT_KNEE, _LM_RIGHT_ANKLE),
    (_LM_NOSE, _LM_LEFT_SHOULDER),
    (_LM_NOSE, _LM_RIGHT_SHOULDER),
]

# ── Colours ──
_COLOR_SKELETON = (0, 255, 255)   # yellow stickman
_COLOR_JOINT = (0, 0, 255)        # red joints
_COLOR_HAND = (255, 0, 255)       # magenta hands
_COLOR_LABEL = (255, 255, 255)    # white text

# ── State ──
_pose_detector = None
_yolo_model = None
_tracker = None
_track_positions: Dict[str, List[Tuple[float, float]]] = {}
_track_hand_positions: Dict[str, List[Tuple[float, float]]] = {}  # wrist positions
_MAX_STALE_TRACKS = 200

# Track exit state: track_id -> {"shelf_id", "baseline_count", "dwell", "entered_at"}
_track_shelf_visits: Dict[str, Dict] = {}

# ── Model file for PoseLandmarker ──
_POSE_MODEL_FILENAME = "pose_landmarker_lite.task"
_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)


def _get_model_path() -> str:
    """Return path to the pose landmarker model, downloading if needed."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(project_root, _POSE_MODEL_FILENAME)
    if not os.path.exists(model_path):
        logger.info("Downloading PoseLandmarker model to %s …", model_path)
        urllib.request.urlretrieve(_POSE_MODEL_URL, model_path)
        logger.info("Download complete (%d bytes).", os.path.getsize(model_path))
    return model_path


def _load_pose_detector():
    """Load MediaPipe PoseLandmarker (Tasks API, once)."""
    global _pose_detector
    if _pose_detector is None:
        import mediapipe as mp

        model_path = _get_model_path()
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,                         # one pose per crop
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _pose_detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        logger.info("MediaPipe PoseLandmarker loaded (lite model).")
    return _pose_detector


def _load_yolo_model():
    """Load the YOLOv8 person detector (once)."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        from shared.config import MODEL_PATHS
        _yolo_model = YOLO(MODEL_PATHS["person"])
        logger.info("YOLOv8 person model loaded for skeleton tracking.")
    return _yolo_model


def _get_tracker():
    """Initialise DeepSort tracker (once)."""
    global _tracker
    if _tracker is None:
        from deep_sort_realtime.deepsort_tracker import DeepSort
        _tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)
        logger.info("DeepSort tracker initialised for skeleton tracking.")
    return _tracker


def _extract_persons_from_pose(
    frame: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Run YOLOv8 for person detection + MediaPipe PoseLandmarker on each crop.

    Returns list of dicts, each with:
        - bbox: [x1, y1, x2, y2]
        - centre: (cx, cy)
        - landmarks: list of (x_px, y_px, visibility) for all 33 landmarks
        - left_wrist: (x, y) or None
        - right_wrist: (x, y) or None
        - confidence: float
    """
    import mediapipe as mp

    yolo = _load_yolo_model()
    pose = _load_pose_detector()

    # Step 1: Detect persons with YOLOv8 (fast, multi-person)
    results = yolo(frame, verbose=False, imgsz=640)
    person_crops = []

    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) != 0:  # person class
                continue
            conf = float(box.conf[0])
            if conf < PERSON_CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            person_crops.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
            })

    # Step 2: Run pose estimation on each person crop
    h, w = frame.shape[:2]
    persons = []

    for crop_info in person_crops:
        x1, y1, x2, y2 = crop_info["bbox"]
        # Add padding to crop for better pose estimation
        pad = 10
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w, x2 + pad)
        cy2 = min(h, y2 + pad)

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        # PoseLandmarker expects RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
        pose_result = pose.detect(mp_image)

        landmarks_px = []
        left_wrist = None
        right_wrist = None

        if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
            crop_h, crop_w = crop.shape[:2]
            # pose_landmarks is a list of pose results; take the first one
            for i, lm in enumerate(pose_result.pose_landmarks[0]):
                # Landmarks are normalised [0,1] relative to the crop
                lx = int(lm.x * crop_w) + cx1
                ly = int(lm.y * crop_h) + cy1
                vis = lm.visibility if hasattr(lm, 'visibility') else lm.presence
                landmarks_px.append((lx, ly, vis))

                if i == _LM_LEFT_WRIST and vis > 0.5:
                    left_wrist = (lx, ly)
                elif i == _LM_RIGHT_WRIST and vis > 0.5:
                    right_wrist = (lx, ly)

        # Body centre = midpoint of shoulders (more stable than bbox centre)
        if len(landmarks_px) > _LM_RIGHT_SHOULDER:
            ls = landmarks_px[_LM_LEFT_SHOULDER]
            rs = landmarks_px[_LM_RIGHT_SHOULDER]
            cx = (ls[0] + rs[0]) / 2
            cy = (ls[1] + rs[1]) / 2
        else:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

        persons.append({
            "bbox": crop_info["bbox"],
            "centre": (cx, cy),
            "landmarks": landmarks_px,
            "left_wrist": left_wrist,
            "right_wrist": right_wrist,
            "confidence": crop_info["confidence"],
        })

    return persons


def detect_customers_skeleton(
    frame: np.ndarray,
    shelf_counts: Dict[str, int] | None = None,
    shelf_regions: List[Dict] | None = None,
) -> Tuple[Dict[str, Any], List[Dict]]:
    """
    Detect and track customers using skeleton pose estimation.

    Args:
        frame:          BGR video frame.
        shelf_counts:   {shelf_id: item_count} from inventory detection.
        shelf_regions:  List of shelf region dicts (auto-detected or manual).

    Returns:
        Tuple of:
        - dict matching CustomerBehaviorData schema
        - list of person dicts with bbox, landmarks, etc. for display
    """
    if shelf_counts is None:
        shelf_counts = {}
    if shelf_regions is None:
        shelf_regions = []

    if MOCK_MODE:
        from vision.customer_tracking import _mock_customers
        return _mock_customers(), []

    tracker = _get_tracker()
    persons = _extract_persons_from_pose(frame)

    # Build DeepSort detections: ([left, top, w, h], confidence, class)
    raw_detections = []
    for p in persons:
        x1, y1, x2, y2 = p["bbox"]
        raw_detections.append(([x1, y1, x2 - x1, y2 - y1], p["confidence"], "person"))

    tracks = tracker.update_tracks(raw_detections, frame=frame)

    customers: List[CustomerItem] = []
    person_display: List[Dict] = []

    # Map track positions to persons by bbox overlap
    track_person_map: Dict[str, Dict] = {}
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = str(track.track_id)
        ltrb = track.to_ltrb()
        tcx = (ltrb[0] + ltrb[2]) / 2
        tcy = (ltrb[1] + ltrb[3]) / 2

        # Find closest person detection
        best_person = None
        best_dist = float("inf")
        for p in persons:
            pcx, pcy = p["centre"]
            d = ((tcx - pcx) ** 2 + (tcy - pcy) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_person = p
        track_person_map[track_id] = best_person or {"bbox": list(ltrb), "centre": (tcx, tcy),
                                                      "landmarks": [], "left_wrist": None,
                                                      "right_wrist": None, "confidence": 0.5}

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = str(track.track_id)
        person = track_person_map.get(track_id)
        if person is None:
            continue

        cx, cy = person["centre"]
        position = (cx, cy)

        # Record position history
        if track_id not in _track_positions:
            _track_positions[track_id] = []
        _track_positions[track_id].append(position)
        _track_positions[track_id] = _track_positions[track_id][-60:]

        # Record hand positions for interaction analysis
        for wrist_key in ("left_wrist", "right_wrist"):
            wrist = person.get(wrist_key)
            if wrist:
                if track_id not in _track_hand_positions:
                    _track_hand_positions[track_id] = []
                _track_hand_positions[track_id].append(wrist)
                _track_hand_positions[track_id] = _track_hand_positions[track_id][-30:]

        # Determine shelf from body centre
        shelf_id = _get_shelf_for_position(cx, cy, shelf_regions)

        # Also check if hand is in a shelf region (stronger pickup signal)
        hand_shelf = None
        for wrist_key in ("left_wrist", "right_wrist"):
            wrist = person.get(wrist_key)
            if wrist:
                hs = _get_shelf_for_position(wrist[0], wrist[1], shelf_regions)
                if hs != "shelf_unknown":
                    hand_shelf = hs
                    break

        # Dwell time (based on body centre)
        dwell = calculate_dwell_time(track_id, position, shelf_regions)

        # Interaction detection (exit-based)
        shelf_count = shelf_counts.get(shelf_id, 0)
        interaction = detect_interaction_on_exit(
            track_id=track_id,
            shelf_id=shelf_id,
            current_shelf_count=shelf_count,
            dwell_seconds=dwell,
            hand_in_shelf=hand_shelf is not None,
        )

        customers.append(
            CustomerItem(
                customer_id=f"cust_{track_id}",
                shelf_id=shelf_id,
                dwell_time_seconds=dwell,
                interaction=interaction,
            )
        )

        person_display.append({
            "bbox": person["bbox"],
            "landmarks": person["landmarks"],
            "confidence": person["confidence"],
            "track_id": track_id,
            "shelf_id": shelf_id,
            "dwell": dwell,
            "interaction": interaction,
            "hand_shelf": hand_shelf,
        })

    behaviour = CustomerBehaviorData(
        schema_version="1.0",
        timestamp=get_current_timestamp(),
        camera_id=CAMERA_ID,
        customers=customers,
    )

    # Cleanup stale tracks
    active_ids = {str(t.track_id) for t in tracks if t.is_confirmed()}
    for store in (_track_positions, _track_hand_positions):
        stale = [tid for tid in store if tid not in active_ids]
        for tid in stale:
            del store[tid]
        if len(store) > _MAX_STALE_TRACKS:
            oldest = sorted(store, key=lambda t: len(store[t]))
            for tid in oldest[:len(store) - _MAX_STALE_TRACKS]:
                del store[tid]

    logger.info("Skeleton tracking: %d customers detected.", len(customers))
    return behaviour.model_dump(), person_display


def _get_shelf_for_position(
    cx: float, cy: float, shelf_regions: List[Dict]
) -> str:
    """Return shelf_id containing the point, or 'shelf_unknown'."""
    for region in shelf_regions:
        x_min = region.get("x_min", 0)
        x_max = region.get("x_max", 0)
        y_min = region.get("y_min", 0)
        y_max = region.get("y_max", 0)
        if x_min <= cx <= x_max and y_min <= cy <= y_max:
            return region.get("shelf_id") or region.get("name", "shelf_unknown")
    return "shelf_unknown"


def draw_skeleton(
    frame: np.ndarray,
    person_display: List[Dict],
) -> np.ndarray:
    """
    Draw stickman skeletons on the frame instead of bounding boxes.

    Args:
        frame: BGR image to draw on (modified in-place).
        person_display: List of person dicts from detect_customers_skeleton().

    Returns:
        The annotated frame.
    """
    for person in person_display:
        landmarks = person.get("landmarks", [])

        if landmarks:
            # Draw skeleton connections
            for (a, b) in _SKELETON_CONNECTIONS:
                if a < len(landmarks) and b < len(landmarks):
                    ax, ay, av = landmarks[a]
                    bx, by, bv = landmarks[b]
                    if av > 0.4 and bv > 0.4:
                        cv2.line(frame, (ax, ay), (bx, by), _COLOR_SKELETON, 2)

            # Draw joints
            for i, (lx, ly, vis) in enumerate(landmarks):
                if vis > 0.4:
                    color = _COLOR_HAND if i in (_LM_LEFT_WRIST, _LM_RIGHT_WRIST) else _COLOR_JOINT
                    radius = 5 if i in (_LM_LEFT_WRIST, _LM_RIGHT_WRIST) else 3
                    cv2.circle(frame, (lx, ly), radius, color, -1)

        # Draw label above head
        track_id = person.get("track_id", "?")
        shelf_id = person.get("shelf_id", "")
        dwell = person.get("dwell", 0)
        interaction = person.get("interaction", "none")

        # Position label at nose or top of bbox
        if landmarks and len(landmarks) > _LM_NOSE and landmarks[_LM_NOSE][2] > 0.4:
            label_x, label_y = landmarks[_LM_NOSE][0], landmarks[_LM_NOSE][1] - 20
        else:
            bbox = person["bbox"]
            label_x = int(bbox[0])
            label_y = int(bbox[1]) - 10

        label = f"ID:{track_id} {shelf_id} {dwell}s"
        if interaction != "none":
            label += f" [{interaction}]"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (label_x, label_y - th - 4),
                      (label_x + tw, label_y + 2), (0, 0, 0), -1)
        cv2.putText(frame, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLOR_LABEL, 1)

        # Highlight hand if it's reaching into a shelf
        if person.get("hand_shelf"):
            for wrist_key in ("left_wrist", "right_wrist"):
                # Find from landmarks
                idx = _LM_LEFT_WRIST if wrist_key == "left_wrist" else _LM_RIGHT_WRIST
                if idx < len(landmarks) and landmarks[idx][2] > 0.4:
                    wx, wy = landmarks[idx][0], landmarks[idx][1]
                    cv2.circle(frame, (wx, wy), 10, _COLOR_HAND, 2)
                    cv2.putText(frame, "REACH", (wx + 12, wy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, _COLOR_HAND, 1)

    return frame
