"""Shared quadrilateral ordering, validation, and refinement helpers."""

import cv2
import numpy as np

from core.config import (
    CARD_CORNER_MAX_SHIFT_RATIO,
    CARD_CORNER_REFINEMENT_WINDOW,
    CARD_RATIO_TOLERANCE,
    CARD_RECOVERY_BORDER_MARGIN_RATIO,
    EXPECTED_CARD_RATIO,
    MIN_IMPROVED_CARD_AREA_RATIO,
)


def order_points(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    reshaped_points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if reshaped_points.shape != (4, 2):
        raise ValueError("Perspektif düzeltme için tam olarak dört nokta gerekli.")

    ordered = np.zeros((4, 2), dtype=np.float32)
    coordinate_sums = reshaped_points.sum(axis=1)
    coordinate_differences = np.diff(reshaped_points, axis=1).reshape(-1)

    ordered[0] = reshaped_points[np.argmin(coordinate_sums)]
    ordered[1] = reshaped_points[np.argmin(coordinate_differences)]
    ordered[2] = reshaped_points[np.argmax(coordinate_sums)]
    ordered[3] = reshaped_points[np.argmax(coordinate_differences)]
    return ordered


def is_card_candidate_valid(
    frame: np.ndarray,
    points: np.ndarray,
) -> bool:
    """Reject geometrically implausible candidates before perspective warping."""
    ordered = order_points(points)
    frame_height, frame_width = frame.shape[:2]
    frame_area = float(frame_height * frame_width)
    if frame_area <= 0 or not cv2.isContourConvex(ordered.astype(np.float32)):
        return False
    area_ratio = abs(float(cv2.contourArea(ordered))) / frame_area
    if not (MIN_IMPROVED_CARD_AREA_RATIO <= area_ratio <= 0.75):
        return False

    widths = (
        float(np.linalg.norm(ordered[1] - ordered[0])),
        float(np.linalg.norm(ordered[2] - ordered[3])),
    )
    heights = (
        float(np.linalg.norm(ordered[3] - ordered[0])),
        float(np.linalg.norm(ordered[2] - ordered[1])),
    )
    average_width = sum(widths) / 2.0
    average_height = sum(heights) / 2.0
    shorter_side = min(average_width, average_height)
    if shorter_side <= 0:
        return False
    card_ratio = max(average_width, average_height) / shorter_side
    return abs(card_ratio - EXPECTED_CARD_RATIO) <= CARD_RATIO_TOLERANCE


def refine_card_corners(frame: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Refine detected corners locally while limiting unsafe corner movement."""
    ordered = order_points(points)
    gray = (
        frame
        if frame.ndim == 2
        else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    )
    widths = [
        np.linalg.norm(ordered[1] - ordered[0]),
        np.linalg.norm(ordered[2] - ordered[3]),
    ]
    heights = [
        np.linalg.norm(ordered[3] - ordered[0]),
        np.linalg.norm(ordered[2] - ordered[1]),
    ]
    shorter_side = float(min(np.mean(widths), np.mean(heights)))
    if shorter_side <= 0:
        return ordered

    window = max(3, int(CARD_CORNER_REFINEMENT_WINDOW))
    corners = ordered.reshape(-1, 1, 2).astype(np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.01,
    )
    try:
        refined = cv2.cornerSubPix(
            gray,
            corners.copy(),
            (window, window),
            (-1, -1),
            criteria,
        ).reshape(4, 2)
    except cv2.error:
        return ordered

    maximum_shift = max(2.0, shorter_side * CARD_CORNER_MAX_SHIFT_RATIO)
    if float(np.max(np.linalg.norm(refined - ordered, axis=1))) > maximum_shift:
        return ordered
    refined = order_points(refined)
    return refined if is_card_candidate_valid(frame, refined) else ordered


def calculate_card_candidate_geometry(
    frame: np.ndarray,
    points: np.ndarray,
) -> tuple[float, float, bool]:
    """Return area ratio, aspect ratio, and opposite-border contact."""
    ordered = order_points(points)
    frame_height, frame_width = frame.shape[:2]
    frame_area = float(frame_height * frame_width)
    area_ratio = (
        abs(float(cv2.contourArea(ordered))) / frame_area
        if frame_area > 0
        else 0.0
    )
    widths = [
        float(np.linalg.norm(ordered[1] - ordered[0])),
        float(np.linalg.norm(ordered[2] - ordered[3])),
    ]
    heights = [
        float(np.linalg.norm(ordered[3] - ordered[0])),
        float(np.linalg.norm(ordered[2] - ordered[1])),
    ]
    shorter_side = min(float(np.mean(widths)), float(np.mean(heights)))
    card_ratio = (
        max(float(np.mean(widths)), float(np.mean(heights))) / shorter_side
        if shorter_side > 0
        else 0.0
    )
    margin = max(3.0, min(frame_height, frame_width) * CARD_RECOVERY_BORDER_MARGIN_RATIO)
    touches_left = float(np.min(ordered[:, 0])) <= margin
    touches_right = float(np.max(ordered[:, 0])) >= frame_width - 1 - margin
    touches_top = float(np.min(ordered[:, 1])) <= margin
    touches_bottom = float(np.max(ordered[:, 1])) >= frame_height - 1 - margin
    touches_opposite_borders = (
        (touches_left and touches_right) or (touches_top and touches_bottom)
    )
    return float(area_ratio), float(card_ratio), touches_opposite_borders
