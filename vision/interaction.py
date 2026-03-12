import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Internal state: track_id -> list of recent (cx, cy) positions
# ──────────────────────────────────────────────

_position_history: Dict[str, List[Tuple[float, float]]] = {}

# How many recent positions to keep per track
_MAX_HISTORY = 30

# Vertical movement threshold (pixels) to classify a pick / replace gesture
_VERTICAL_MOVE_THRESHOLD = 40.0


def detect_interaction(track_id: str, positions: List[Tuple[float, float]]) -> str:
    """
    Determine the interaction type for a tracked person.

    The function analyses the recent trajectory of the person's bounding-box
    centre to infer a shelf interaction:

    * **picked_product** – the person's hand/body moved *upward* toward the
      shelf and then *away* (a noticeable upward-then-downward Y pattern).
    * **replaced_product** – the opposite pattern (downward-then-upward),
      suggesting the person put something back.
    * **none** – no significant vertical movement detected.

    Args:
        track_id:   Unique identifier for the tracked person.
        positions:  List of ``(centre_x, centre_y)`` coordinates collected
                    so far for this track (most recent last).

    Returns:
        One of ``"none"``, ``"picked_product"``, or ``"replaced_product"``.
    """
    # Update internal history
    if track_id not in _position_history:
        _position_history[track_id] = []

    _position_history[track_id].extend(positions)
    # Keep only the most recent positions
    _position_history[track_id] = _position_history[track_id][-_MAX_HISTORY:]

    history = _position_history[track_id]

    if len(history) < 6:
        return "none"

    # Split history into two halves and compare average Y values.
    # In image coordinates, Y increases downward:
    #   • moving hand UP toward shelf  → Y decreases then increases  → "picked_product"
    #   • moving hand DOWN to shelf    → Y increases then decreases  → "replaced_product"
    mid = len(history) // 2
    first_half_y = sum(p[1] for p in history[:mid]) / mid
    second_half_y = sum(p[1] for p in history[mid:]) / (len(history) - mid)

    delta_y = second_half_y - first_half_y

    if delta_y > _VERTICAL_MOVE_THRESHOLD:
        # Y increased → hand went down after going up → picked product
        interaction = "picked_product"
    elif delta_y < -_VERTICAL_MOVE_THRESHOLD:
        # Y decreased → hand went up after going down → replaced product
        interaction = "replaced_product"
    else:
        interaction = "none"

    logger.debug("Track %s interaction: %s (delta_y=%.1f)", track_id, interaction, delta_y)
    return interaction


def reset() -> None:
    """Clear all accumulated position history (useful for tests)."""
    _position_history.clear()
