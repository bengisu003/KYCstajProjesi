"""Primary contour, brightness, and internal-content card detectors."""

import cv2
import numpy as np

from core.config import (
    CARD_DETECTION_MAX_DIMENSION,
    CARD_RATIO_TOLERANCE,
    EXPECTED_CARD_RATIO,
    MAX_FALLBACK_CARD_AREA_RATIO,
    MAX_INTERNAL_CARD_AREA_RATIO,
    MIN_CARD_AREA_RATIO,
    MIN_INTERNAL_CARD_AREA_RATIO,
)
from infrastructure.vision.card.fallback_detection import (
    _detect_card_from_multichannel_fallback,
)
from infrastructure.vision.card.geometry import order_points


def _detect_card_baseline(frame: np.ndarray) -> np.ndarray | None:
    """Detect the best card-shaped quadrilateral in an image frame."""
    if frame.size == 0 or frame.ndim not in (2, 3):
        return None

    original_height, original_width = frame.shape[:2]
    longest_side = max(original_height, original_width)
    detection_scale = min(1.0, CARD_DETECTION_MAX_DIMENSION / longest_side)
    if detection_scale < 1.0:
        detection_frame = cv2.resize(
            frame,
            (
                max(1, int(round(original_width * detection_scale))),
                max(1, int(round(original_height * detection_scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    else:
        detection_frame = frame

    if detection_frame.ndim == 2:
        gray = detection_frame
    else:
        gray = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel_size = max(5, int(round(min(gray.shape[:2]) * 0.008)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    closing_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    closed_edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        closing_kernel,
        iterations=2,
    )

    contours, _ = cv2.findContours(
        closed_edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    frame_area = float(gray.shape[0] * gray.shape[1])
    if frame_area <= 0:
        return None

    best_points: np.ndarray | None = None
    best_score = -1.0

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        contour_area = float(cv2.contourArea(contour))
        area_ratio = contour_area / frame_area
        if area_ratio < MIN_CARD_AREA_RATIO:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue

        candidate = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(candidate) != 4 or not cv2.isContourConvex(candidate):
            continue

        ordered = order_points(candidate)
        top_width = float(np.linalg.norm(ordered[1] - ordered[0]))
        bottom_width = float(np.linalg.norm(ordered[2] - ordered[3]))
        left_height = float(np.linalg.norm(ordered[3] - ordered[0]))
        right_height = float(np.linalg.norm(ordered[2] - ordered[1]))

        average_width = (top_width + bottom_width) / 2.0
        average_height = (left_height + right_height) / 2.0
        shorter_side = min(average_width, average_height)
        longer_side = max(average_width, average_height)
        if shorter_side <= 0:
            continue

        card_ratio = longer_side / shorter_side
        ratio_difference = abs(card_ratio - EXPECTED_CARD_RATIO)
        if ratio_difference > CARD_RATIO_TOLERANCE:
            continue

        ratio_score = 1.0 - (ratio_difference / CARD_RATIO_TOLERANCE)
        candidate_score = (0.6 * ratio_score) + (0.4 * min(area_ratio, 1.0))
        if candidate_score > best_score:
            best_score = candidate_score
            best_points = ordered

    if best_points is not None:
        return best_points / detection_scale

    brightness_candidate = _detect_card_from_brightness_mask(gray, frame_area)
    if brightness_candidate is not None:
        return brightness_candidate / detection_scale

    internal_candidate = _detect_card_from_internal_contours(gray, frame_area)
    if internal_candidate is not None:
        return internal_candidate / detection_scale

    # Preserve the original fast detector above. The more expensive fallback
    # only runs when every existing path fails, so successful baseline crops
    # and the downstream classification score remain unchanged.
    return _detect_card_from_multichannel_fallback(frame)




def _detect_card_from_brightness_mask(
    gray: np.ndarray,
    frame_area: float,
) -> np.ndarray | None:
    """Find a card rectangle from a brightness mask when edges are fragmented."""
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, brightness_mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    closed_mask = cv2.morphologyEx(
        brightness_mask,
        cv2.MORPH_CLOSE,
        closing_kernel,
        iterations=2,
    )
    contours, _ = cv2.findContours(
        closed_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    best_points: np.ndarray | None = None
    best_score = -1.0

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        contour_area_ratio = float(cv2.contourArea(contour)) / frame_area
        if contour_area_ratio < MIN_CARD_AREA_RATIO / 2.0:
            continue

        rectangle = cv2.minAreaRect(contour)
        candidate = order_points(cv2.boxPoints(rectangle))
        candidate_area_ratio = abs(float(cv2.contourArea(candidate))) / frame_area
        if not (
            MIN_CARD_AREA_RATIO
            <= candidate_area_ratio
            <= MAX_FALLBACK_CARD_AREA_RATIO
        ):
            continue

        top_width = float(np.linalg.norm(candidate[1] - candidate[0]))
        bottom_width = float(np.linalg.norm(candidate[2] - candidate[3]))
        left_height = float(np.linalg.norm(candidate[3] - candidate[0]))
        right_height = float(np.linalg.norm(candidate[2] - candidate[1]))
        average_width = (top_width + bottom_width) / 2.0
        average_height = (left_height + right_height) / 2.0
        shorter_side = min(average_width, average_height)
        if shorter_side <= 0:
            continue

        card_ratio = max(average_width, average_height) / shorter_side
        ratio_difference = abs(card_ratio - EXPECTED_CARD_RATIO)
        if ratio_difference > CARD_RATIO_TOLERANCE:
            continue

        ratio_score = 1.0 - ratio_difference / CARD_RATIO_TOLERANCE
        candidate_score = 0.7 * ratio_score + 0.3 * candidate_area_ratio
        if candidate_score > best_score:
            best_score = candidate_score
            best_points = candidate

    return best_points




def _detect_card_from_internal_contours(
    gray: np.ndarray,
    frame_area: float,
) -> np.ndarray | None:
    """Detect a printed card region nested inside a larger paper contour."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive_mask = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        7,
    )
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed_mask = cv2.morphologyEx(
        adaptive_mask,
        cv2.MORPH_CLOSE,
        closing_kernel,
        iterations=2,
    )
    contours, _ = cv2.findContours(
        closed_mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:100]:
        rectangle = cv2.minAreaRect(contour)
        candidate = order_points(cv2.boxPoints(rectangle))
        candidate_area_ratio = abs(float(cv2.contourArea(candidate))) / frame_area
        if not (
            MIN_INTERNAL_CARD_AREA_RATIO
            <= candidate_area_ratio
            <= MAX_INTERNAL_CARD_AREA_RATIO
        ):
            continue

        top_width = float(np.linalg.norm(candidate[1] - candidate[0]))
        bottom_width = float(np.linalg.norm(candidate[2] - candidate[3]))
        left_height = float(np.linalg.norm(candidate[3] - candidate[0]))
        right_height = float(np.linalg.norm(candidate[2] - candidate[1]))
        average_width = (top_width + bottom_width) / 2.0
        average_height = (left_height + right_height) / 2.0
        shorter_side = min(average_width, average_height)
        if shorter_side <= 0:
            continue

        card_ratio = max(average_width, average_height) / shorter_side
        if abs(card_ratio - EXPECTED_CARD_RATIO) <= CARD_RATIO_TOLERANCE:
            return candidate

    return None
