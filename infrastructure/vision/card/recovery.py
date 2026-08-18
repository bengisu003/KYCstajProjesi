"""Recover suspicious card candidates using color and front-flag evidence."""

import cv2
import numpy as np

from core.config import (
    CARD_DETECTION_MAX_DIMENSION,
    CARD_FLAG_ANCHOR_CENTER_X_RATIO,
    CARD_FLAG_ANCHOR_CENTER_Y_RATIO,
    CARD_FLAG_ANCHOR_HEIGHT_SCALE,
    CARD_FLAG_ANCHOR_MIN_COMPONENT_AREA_RATIO,
    CARD_FLAG_ANCHOR_WIDTH_SCALE,
    CARD_RATIO_TOLERANCE,
    CARD_RECOVERY_FRONT_RED_RATIO,
    CARD_RECOVERY_LARGE_AREA_MAX_RATIO,
    CARD_RECOVERY_LARGE_AREA_RATIO,
    CARD_RECOVERY_MAX_RATIO,
    CARD_RECOVERY_MIN_RATIO,
    CARD_RECOVERY_TINY_AREA_RATIO,
    EXPECTED_CARD_RATIO,
    MIN_CARD_AREA_RATIO,
)
from infrastructure.vision.card.geometry import (
    calculate_card_candidate_geometry,
    is_card_candidate_valid,
    order_points,
)
from infrastructure.vision.card.transformation import warp_card


def _detect_card_from_lab_color_mask(frame: np.ndarray) -> np.ndarray | None:
    """Find a cool-toned card region against a warmer paper/table background."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        return None

    height, width = frame.shape[:2]
    longest_side = max(height, width)
    scale = min(1.0, CARD_DETECTION_MAX_DIMENSION / longest_side)
    if scale < 1.0:
        detection_frame = cv2.resize(
            frame,
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    else:
        detection_frame = frame

    lab = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2LAB)
    _threshold, color_mask = cv2.threshold(
        lab[:, :, 2],
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    mask_height, mask_width = color_mask.shape
    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            max(15, int(round(mask_width * 0.03))),
            max(9, int(round(mask_height * 0.02))),
        ),
    )
    closed_mask = cv2.morphologyEx(
        color_mask,
        cv2.MORPH_CLOSE,
        closing_kernel,
        iterations=2,
    )
    contours, _ = cv2.findContours(
        closed_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    frame_area = float(mask_height * mask_width)
    best_points: np.ndarray | None = None
    best_score = -1.0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        candidate = order_points(cv2.boxPoints(cv2.minAreaRect(contour)))
        candidate_area = abs(float(cv2.contourArea(candidate)))
        area_ratio = candidate_area / frame_area if frame_area > 0 else 0.0
        if not (MIN_CARD_AREA_RATIO * 0.75 <= area_ratio <= 0.75):
            continue

        widths = [
            float(np.linalg.norm(candidate[1] - candidate[0])),
            float(np.linalg.norm(candidate[2] - candidate[3])),
        ]
        heights = [
            float(np.linalg.norm(candidate[3] - candidate[0])),
            float(np.linalg.norm(candidate[2] - candidate[1])),
        ]
        shorter_side = min(float(np.mean(widths)), float(np.mean(heights)))
        if shorter_side <= 0:
            continue
        card_ratio = max(float(np.mean(widths)), float(np.mean(heights))) / shorter_side
        ratio_difference = abs(card_ratio - EXPECTED_CARD_RATIO)
        if ratio_difference > CARD_RATIO_TOLERANCE:
            continue

        rectangularity = float(
            np.clip(abs(float(cv2.contourArea(contour))) / max(candidate_area, 1.0), 0.0, 1.0)
        )
        ratio_score = 1.0 - ratio_difference / CARD_RATIO_TOLERANCE
        area_score = float(np.clip(area_ratio / 0.35, 0.0, 1.0))
        score = 0.65 * ratio_score + 0.15 * area_score + 0.20 * rectangularity
        if score > best_score:
            best_score = score
            best_points = candidate

    return best_points / scale if best_points is not None else None


def _should_try_color_recovery(
    frame: np.ndarray,
    points: np.ndarray,
) -> bool:
    """Return whether the initial contour has suspicious crop geometry."""
    area_ratio, card_ratio, opposite_borders = calculate_card_candidate_geometry(
        frame, points
    )
    return bool(
        area_ratio < CARD_RECOVERY_TINY_AREA_RATIO
        or card_ratio < CARD_RECOVERY_MIN_RATIO
        or card_ratio > CARD_RECOVERY_MAX_RATIO
        or opposite_borders
        or (
            area_ratio > CARD_RECOVERY_LARGE_AREA_RATIO
            and card_ratio < CARD_RECOVERY_LARGE_AREA_MAX_RATIO
        )
    )


def _is_better_color_recovery(
    frame: np.ndarray,
    original: np.ndarray,
    recovered: np.ndarray,
) -> bool:
    """Accept a recovery only when it clearly improves suspicious geometry."""
    original_geometry = calculate_card_candidate_geometry(frame, original)
    recovered_geometry = calculate_card_candidate_geometry(frame, recovered)
    original_area, original_ratio, original_borders = original_geometry
    recovered_area, recovered_ratio, recovered_borders = recovered_geometry
    if original_area < CARD_RECOVERY_TINY_AREA_RATIO and recovered_area >= 0.12:
        return True
    if original_borders and not recovered_borders and recovered_area >= 0.10:
        return True
    if original_borders and recovered_area < original_area * 0.97:
        return True
    return bool(
        abs(recovered_ratio - EXPECTED_CARD_RATIO) + 0.08
        < abs(original_ratio - EXPECTED_CARD_RATIO)
    )


def _looks_like_large_rotated_front_crop(
    frame: np.ndarray,
    points: np.ndarray,
) -> bool:
    """Detect a front card left small/rotated inside an oversized paper crop."""
    area_ratio, card_ratio, _opposite_borders = calculate_card_candidate_geometry(
        frame, points
    )
    if not (
        area_ratio > CARD_RECOVERY_LARGE_AREA_RATIO
        and card_ratio < CARD_RECOVERY_LARGE_AREA_MAX_RATIO
    ):
        return False
    warped = warp_card(frame, points)
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    red_mask = (
        ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 170))
        & (hsv[:, :, 1] > 35)
        & (hsv[:, :, 2] > 60)
    )
    return float(np.mean(red_mask)) >= CARD_RECOVERY_FRONT_RED_RATIO


def _detect_front_card_from_flag_anchor(frame: np.ndarray) -> np.ndarray | None:
    """Estimate a small printed front card from its red flag components.

    This deliberately narrow fallback is used only when the ordinary detector
    has selected an oversized paper region containing front-side flag evidence.
    It does not participate in normal card detection or back-side detection.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        return None

    height, width = frame.shape[:2]
    longest_side = max(height, width)
    scale = min(1.0, CARD_DETECTION_MAX_DIMENSION / longest_side)
    if scale < 1.0:
        detection_frame = cv2.resize(
            frame,
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    else:
        detection_frame = frame

    lab = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2LAB)
    red_axis = lab[:, :, 1]
    red_threshold = max(140.0, float(np.median(red_axis)) + 12.0)
    red_mask = np.where(red_axis >= red_threshold, 255, 0).astype(np.uint8)
    mask_height, mask_width = red_mask.shape
    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            max(7, int(round(mask_width * 0.012))),
            max(5, int(round(mask_height * 0.008))),
        ),
    )
    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_CLOSE,
        closing_kernel,
        iterations=1,
    )
    contours, _ = cv2.findContours(
        red_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    frame_area = float(mask_height * mask_width)
    components = [
        contour
        for contour in contours
        if cv2.contourArea(contour)
        >= frame_area * CARD_FLAG_ANCHOR_MIN_COMPONENT_AREA_RATIO
    ]
    if not components:
        return None

    crescent = max(components, key=cv2.contourArea)
    crescent_area = float(cv2.contourArea(crescent))
    x, y, component_width, component_height = cv2.boundingRect(crescent)
    crescent_center = np.array(
        [x + component_width / 2.0, y + component_height / 2.0],
        dtype=np.float32,
    )
    flag_components = [crescent]
    for contour in components:
        if contour is crescent:
            continue
        contour_area = float(cv2.contourArea(contour))
        other_x, other_y, other_width, other_height = cv2.boundingRect(contour)
        other_center = np.array(
            [other_x + other_width / 2.0, other_y + other_height / 2.0],
            dtype=np.float32,
        )
        offset = np.abs(other_center - crescent_center)
        if (
            contour_area >= crescent_area * 0.15
            and offset[0] <= component_width * 1.6
            and offset[1] <= component_height * 0.9
        ):
            flag_components.append(contour)

    combined = np.vstack(flag_components)
    flag_x, flag_y, flag_width, flag_height = cv2.boundingRect(combined)
    if flag_width <= 0 or flag_height <= 0:
        return None

    estimated_width = flag_width * CARD_FLAG_ANCHOR_WIDTH_SCALE
    estimated_height = flag_height * CARD_FLAG_ANCHOR_HEIGHT_SCALE
    flag_center_x = flag_x + flag_width / 2.0
    flag_center_y = flag_y + flag_height / 2.0
    left = flag_center_x - CARD_FLAG_ANCHOR_CENTER_X_RATIO * estimated_width
    top = flag_center_y - CARD_FLAG_ANCHOR_CENTER_Y_RATIO * estimated_height
    right = left + estimated_width
    bottom = top + estimated_height

    candidate = np.array(
        [[left, top], [right, top], [right, bottom], [left, bottom]],
        dtype=np.float32,
    )
    candidate[:, 0] = np.clip(candidate[:, 0], 0, mask_width - 1)
    candidate[:, 1] = np.clip(candidate[:, 1], 0, mask_height - 1)
    candidate /= scale
    return candidate if is_card_candidate_valid(frame, candidate) else None
