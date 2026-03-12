import logging
import math
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Multi-signal interaction detection  (v3 — hand gesture + dwell time)
#
# Key insight: product count delta alone is unreliable because items
# behind the picked item fill the visual gap → count stays the same.
#
# PRIMARY SIGNALS (highest reliability):
#   1. Hand reaching INTO shelf region (wrist keypoint in shelf bbox)
#   2. Grab motion: hand enters shelf → pauses → exits shelf
#   3. Dwell time near the shelf
#
# SECONDARY SIGNAL (used as bonus, not relied upon):
#   4. Count delta (baseline vs final item count)
#
# PRODUCT IDENTIFICATION:
#   During a grab event, the wrist position is matched to the nearest
#   product bounding box to identify WHICH item was picked/replaced.
#
# Combined into 4 outcomes:
#   • "picked_product"    — hand grabbed + dwell > 2s (high-confidence buy)
#   • "replaced_product"  — hand reached, count went UP
#   • "interested_no_buy" — stood near shelf > 5s, no grab
#   • "none"              — brief pass, no engagement
# ──────────────────────────────────────────────

# ── Return type for exit-based detection ──
# detect_interaction_on_exit now returns a dict:
#   {"interaction": str, "product_name": str | None}

# ── State for exit-based mode ──
# track_id -> visit state dict
_track_visit_state: Dict[str, Dict] = {}

# track_id -> last resolved result dict
_track_exit_results: Dict[str, Dict] = {}

_MAX_TRACKED_IDS = 300

# ── Thresholds ──
_MIN_DWELL_FOR_PURCHASE = 2.0      # hand reach + dwell > 2s = bought
_MIN_DWELL_FOR_INTEREST = 5.0      # no reach + dwell > 5s = interested but no buy
_MAX_DWELL_BROWSING = 120.0        # beyond this, tracker probably glitched
_MIN_HAND_REACH_FRAMES = 2         # hand must be in shelf region for ≥ 2 frames
_HAND_GRAB_PAUSE_MIN = 0.3         # seconds hand stays in shelf = "grab" vs "swipe"


# ═══════════════════════════════════════════════
# Legacy continuous detection (kept for backward compat)
# ═══════════════════════════════════════════════

# track_id -> {"shelf_id": str, "baseline_count": int, "last_update": float}
_track_shelf_state: Dict[str, Dict] = {}
_MIN_DWELL_FOR_INTERACTION = 1.0

def detect_interaction(
    track_id: str,
    shelf_id: str,
    current_shelf_count: int,
    dwell_seconds: int,
) -> str:
    """
    Determine the interaction type by comparing shelf item counts.
    (Legacy continuous mode — fires every frame.)

    Returns:
        One of ``"none"``, ``"picked_product"``, or ``"replaced_product"``.
    """
    if shelf_id == "shelf_unknown" or dwell_seconds < _MIN_DWELL_FOR_INTERACTION:
        return "none"

    state = _track_shelf_state.get(track_id)

    if state is None or state["shelf_id"] != shelf_id:
        _track_shelf_state[track_id] = {
            "shelf_id": shelf_id,
            "baseline_count": current_shelf_count,
            "last_update": time.time(),
        }
        return "none"

    baseline = state["baseline_count"]
    state["last_update"] = time.time()

    if current_shelf_count < baseline:
        state["baseline_count"] = current_shelf_count
        return "picked_product"

    if current_shelf_count > baseline:
        state["baseline_count"] = current_shelf_count
        return "replaced_product"

    return "none"


# ═══════════════════════════════════════════════
# New multi-signal detection (preferred)
# ═══════════════════════════════════════════════

def detect_interaction_on_exit(
    track_id: str,
    shelf_id: str,
    current_shelf_count: int,
    dwell_seconds: int,
    hand_in_shelf: bool = False,
) -> str:
    """
    Multi-signal interaction detection using hand gestures + dwell time.

    While the customer is AT a shelf → accumulates state silently,
    tracking hand reach events and dwell duration.

    When the customer LEAVES → evaluates signals and returns result.

    Decision matrix:
        hand_grabbed + dwell ≥ 2s         → "picked_product"
        hand_reached + count increased     → "replaced_product"
        no_grab + dwell ≥ 5s              → "interested_no_buy"
        anything else                      → "none"

    Args:
        track_id:            Unique tracked person ID.
        shelf_id:            Current shelf (or "shelf_unknown" if not near any).
        current_shelf_count: Items on the current shelf right now.
        dwell_seconds:       Current dwell time at this shelf.
        hand_in_shelf:       True if a hand keypoint is inside a shelf region.

    Returns:
        One of "none", "picked_product", "replaced_product", "interested_no_buy".
    """
    state = _track_visit_state.get(track_id)
    now = time.time()

    # ── Customer LEFT the shelf (or moved to a different one) ──
    if state is not None and (shelf_id == "shelf_unknown" or shelf_id != state["shelf_id"]):
        result = _evaluate_visit(track_id, state)
        _track_visit_state.pop(track_id, None)
        _track_exit_results[track_id] = result
        # Start tracking new shelf if applicable
        if shelf_id != "shelf_unknown":
            _track_visit_state[track_id] = _new_visit(shelf_id, current_shelf_count, now, hand_in_shelf)
        return result

    # ── Customer is AT a shelf (or just arrived) ──
    if shelf_id == "shelf_unknown":
        pending = _track_exit_results.pop(track_id, None)
        return pending or "none"

    if state is None or state["shelf_id"] != shelf_id:
        _track_visit_state[track_id] = _new_visit(shelf_id, current_shelf_count, now, hand_in_shelf)
        return "none"

    # ── Update running state while dwelling ──
    state["last_count"] = current_shelf_count
    state["last_update"] = now

    if hand_in_shelf:
        state["hand_reach_frames"] += 1
        if state["hand_enter_time"] is None:
            state["hand_enter_time"] = now     # hand just entered shelf
    else:
        # Hand left the shelf region — check if this was a grab
        if state["hand_enter_time"] is not None:
            hand_duration = now - state["hand_enter_time"]
            if hand_duration >= _HAND_GRAB_PAUSE_MIN:
                state["grab_events"] += 1
                logger.debug(
                    "Track %s: grab event #%d at shelf %s (hand in shelf %.1fs)",
                    track_id, state["grab_events"], shelf_id, hand_duration,
                )
            state["hand_enter_time"] = None    # reset for next reach

    return "none"


def _new_visit(shelf_id: str, count: int, now: float, hand_in: bool) -> Dict:
    """Create a fresh visit state dict."""
    return {
        "shelf_id": shelf_id,
        "baseline_count": count,
        "entered_at": now,
        "last_count": count,
        "last_update": now,
        # Hand gesture tracking
        "hand_reach_frames": 1 if hand_in else 0,
        "hand_enter_time": now if hand_in else None,
        "grab_events": 0,       # hand entered shelf, paused, then left = 1 grab
    }


def _evaluate_visit(track_id: str, state: Dict) -> str:
    """
    Evaluate a completed shelf visit using multiple signals.

    Priority:
      1. Grab motion detected (hand entered + paused + exited)
      2. Hand reached into shelf (even without clean grab)
      3. Dwell time alone (no hand data — fallback)
      4. Count delta (secondary bonus signal)
    """
    baseline = state["baseline_count"]
    final_count = state["last_count"]
    dwell = time.time() - state["entered_at"]
    shelf_id = state["shelf_id"]

    grab_events = state.get("grab_events", 0)
    hand_reach_frames = state.get("hand_reach_frames", 0)
    hand_reached = hand_reach_frames >= _MIN_HAND_REACH_FRAMES

    delta = final_count - baseline  # negative = items taken, positive = items added

    # ── Sanity check: abnormally long dwell is likely a stale tracker ──
    if dwell > _MAX_DWELL_BROWSING:
        logger.debug("Track %s: dwell %.0fs at %s — ignoring (likely stale).", track_id, dwell, shelf_id)
        return "none"

    # ── Signal 1: Hand grab detected (strongest signal) ──
    if grab_events > 0 and dwell >= _MIN_DWELL_FOR_PURCHASE:
        logger.info(
            "Track %s EXIT shelf %s: PICKED PRODUCT "
            "(grab_events=%d, dwell=%.1fs, count_delta=%+d)",
            track_id, shelf_id, grab_events, dwell, delta,
        )
        return "picked_product"

    # ── Signal 2: Hand reached in but no clean grab ──
    if hand_reached:
        if delta > 0:
            # Hand reached in and count went up → putting something back
            logger.info(
                "Track %s EXIT shelf %s: REPLACED PRODUCT "
                "(hand_reached, count_delta=%+d, dwell=%.1fs)",
                track_id, shelf_id, delta, dwell,
            )
            return "replaced_product"

        if dwell >= _MIN_DWELL_FOR_PURCHASE:
            # Hand reached in + decent dwell → likely picked something
            logger.info(
                "Track %s EXIT shelf %s: PICKED PRODUCT "
                "(hand_reached=%d frames, dwell=%.1fs, count_delta=%+d)",
                track_id, shelf_id, hand_reach_frames, dwell, delta,
            )
            return "picked_product"

        # Hand entered shelf briefly — quick touch, doesn't count
        logger.debug(
            "Track %s EXIT shelf %s: hand briefly in shelf (%.1fs) → none",
            track_id, shelf_id, dwell,
        )
        return "none"

    # ── Signal 3: No hand data — fall back to dwell time ──
    if dwell >= _MIN_DWELL_FOR_INTEREST:
        # Stood near shelf for a while but never reached in
        logger.info(
            "Track %s EXIT shelf %s: INTERESTED NO BUY "
            "(dwell=%.1fs, no hand reach, count_delta=%+d)",
            track_id, shelf_id, dwell, delta,
        )
        return "interested_no_buy"

    # ── Signal 4: Count delta only (weakest, used as last resort) ──
    if delta < 0 and dwell >= _MIN_DWELL_FOR_PURCHASE:
        logger.info(
            "Track %s EXIT shelf %s: PICKED PRODUCT (count-only fallback, "
            "delta=%+d, dwell=%.1fs, no hand data)",
            track_id, shelf_id, delta, dwell,
        )
        return "picked_product"

    if delta > 0:
        logger.info(
            "Track %s EXIT shelf %s: REPLACED PRODUCT (count-only, delta=%+d)",
            track_id, shelf_id, delta,
        )
        return "replaced_product"

    # Nothing interesting happened
    return "none"


# ═══════════════════════════════════════════════
# Cleanup / reset
# ═══════════════════════════════════════════════

def reset() -> None:
    """Clear all accumulated state (useful for tests)."""
    _track_shelf_state.clear()
    _track_visit_state.clear()
    _track_exit_results.clear()


def cleanup_stale_tracks(active_ids: set) -> None:
    """Remove state for tracks that are no longer active."""
    for store in (_track_shelf_state, _track_visit_state, _track_exit_results):
        stale = [tid for tid in store if tid not in active_ids]
        for tid in stale:
            del store[tid]
    # Hard cap
    if len(_track_visit_state) > _MAX_TRACKED_IDS:
        by_time = sorted(_track_visit_state,
                         key=lambda t: _track_visit_state[t]["last_update"])
        for tid in by_time[:len(_track_visit_state) - _MAX_TRACKED_IDS]:
            del _track_visit_state[tid]
