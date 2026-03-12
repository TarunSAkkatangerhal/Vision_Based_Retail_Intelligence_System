import logging
import time
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Internal state: track_id -> {"enter_time", "shelf_id"}
# ──────────────────────────────────────────────

_track_timers: Dict[str, Dict] = {}


def _point_in_region(
    position: Tuple[float, float],
    region: Dict,
) -> bool:
    """Check whether a (cx, cy) point falls inside a rectangular shelf region."""
    cx, cy = position
    return (
        region["x_min"] <= cx <= region["x_max"]
        and region["y_min"] <= cy <= region["y_max"]
    )


def calculate_dwell_time(
    track_id: str,
    position: Tuple[float, float],
    shelf_regions: List[Dict],
) -> int:
    """
    Return the dwell time **in seconds** for a tracked person near a shelf.

    Args:
        track_id:  Unique identifier for the tracked person.
        position:  (centre_x, centre_y) of the person bounding box.
        shelf_regions: List of dicts, each with keys
            ``shelf_id``, ``x_min``, ``x_max``, ``y_min``, ``y_max``.

    Returns:
        Dwell time in whole seconds.  0 if the person is not currently
        inside any shelf region.
    """
    current_shelf: str | None = None
    for region in shelf_regions:
        if _point_in_region(position, region):
            current_shelf = region["shelf_id"]
            break

    if current_shelf is None:
        # Person left all shelf regions → reset their timer
        if track_id in _track_timers:
            del _track_timers[track_id]
        return 0

    if track_id not in _track_timers:
        # First time we see this person near a shelf
        _track_timers[track_id] = {
            "enter_time": time.time(),
            "shelf_id": current_shelf,
        }
        return 0

    entry = _track_timers[track_id]

    # If the person moved to a *different* shelf, restart the timer
    if entry["shelf_id"] != current_shelf:
        _track_timers[track_id] = {
            "enter_time": time.time(),
            "shelf_id": current_shelf,
        }
        return 0

    dwell = int(time.time() - entry["enter_time"])
    logger.debug(
        "Track %s dwell time at %s: %d s", track_id, current_shelf, dwell
    )
    return dwell


def reset() -> None:
    """Clear all accumulated dwell-time state (useful for tests)."""
    _track_timers.clear()
