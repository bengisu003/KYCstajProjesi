"""Orchestrate baseline detection, refinement, and guarded recovery."""

import numpy as np

from core.config import CARD_DETECTION_MODE
from infrastructure.vision.card.baseline_detection import _detect_card_baseline
from infrastructure.vision.card.geometry import (
    is_card_candidate_valid,
    refine_card_corners,
)
from infrastructure.vision.card.recovery import (
    _detect_card_from_lab_color_mask,
    _detect_front_card_from_flag_anchor,
    _is_better_color_recovery,
    _looks_like_large_rotated_front_crop,
    _should_try_color_recovery,
)


def detect_card(frame: np.ndarray) -> np.ndarray | None:
    """Detect a card using the selectable baseline or improved geometry mode."""
    points = _detect_card_baseline(frame)
    if points is None or CARD_DETECTION_MODE.lower() != "improved":
        return points
    if not is_card_candidate_valid(frame, points):
        return None
    refined_points = refine_card_corners(frame, points)
    if not _should_try_color_recovery(frame, refined_points):
        return refined_points

    recovered_points = _detect_card_from_lab_color_mask(frame)
    if (
        recovered_points is not None
        and is_card_candidate_valid(frame, recovered_points)
    ):
        recovered_points = refine_card_corners(frame, recovered_points)
        if _is_better_color_recovery(frame, refined_points, recovered_points):
            return recovered_points

    # An oversized, rotated front-side paper crop is unsafe to classify. Try a
    # narrow flag-anchored recovery before rejecting it. Back cards do not
    # contain the red flag region used by this recovery and guard.
    if _looks_like_large_rotated_front_crop(frame, refined_points):
        flag_candidate = _detect_front_card_from_flag_anchor(frame)
        if flag_candidate is not None:
            return refine_card_corners(frame, flag_candidate)
        return None
    return refined_points
