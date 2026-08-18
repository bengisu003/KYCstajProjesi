"""Multi-channel and edge-projection fallback card detectors."""

import cv2
import numpy as np

from core.config import (
    CARD_DETECTION_FALLBACK_MAX_DIMENSION,
    CARD_RATIO_TOLERANCE,
    EXPECTED_CARD_RATIO,
    MAX_INTERNAL_CARD_AREA_RATIO,
    MIN_CARD_AREA_RATIO,
)
from infrastructure.vision.card.geometry import order_points

def _score_multichannel_card_candidate(
    candidate: np.ndarray,
    contour: np.ndarray,
    gray: np.ndarray,
    edge_union: np.ndarray,
    frame_area: float,
) -> float | None:
    """Score a fallback quadrilateral using geometry and internal detail."""
    ordered = order_points(candidate)
    candidate_area = abs(float(cv2.contourArea(ordered)))
    if candidate_area <= 0 or frame_area <= 0:
        return None

    area_ratio = candidate_area / frame_area
    if not (MIN_CARD_AREA_RATIO * 0.75 <= area_ratio <= 0.75):
        return None

    # A fallback rectangle spanning two opposite image borders is usually the
    # table/paper background rather than the printed identity-card boundary.
    frame_height, frame_width = gray.shape[:2]
    border_margin = max(3.0, min(frame_height, frame_width) * 0.008)
    touches_left = float(np.min(ordered[:, 0])) <= border_margin
    touches_right = float(np.max(ordered[:, 0])) >= frame_width - 1 - border_margin
    touches_top = float(np.min(ordered[:, 1])) <= border_margin
    touches_bottom = float(np.max(ordered[:, 1])) >= frame_height - 1 - border_margin
    if (touches_left and touches_right) or (touches_top and touches_bottom):
        return None

    top_width = float(np.linalg.norm(ordered[1] - ordered[0]))
    bottom_width = float(np.linalg.norm(ordered[2] - ordered[3]))
    left_height = float(np.linalg.norm(ordered[3] - ordered[0]))
    right_height = float(np.linalg.norm(ordered[2] - ordered[1]))
    average_width = (top_width + bottom_width) / 2.0
    average_height = (left_height + right_height) / 2.0
    shorter_side = min(average_width, average_height)
    if shorter_side <= 0:
        return None

    card_ratio = max(average_width, average_height) / shorter_side
    ratio_difference = abs(card_ratio - EXPECTED_CARD_RATIO)
    if ratio_difference > CARD_RATIO_TOLERANCE:
        return None
    ratio_score = 1.0 - ratio_difference / CARD_RATIO_TOLERANCE

    contour_area = abs(float(cv2.contourArea(contour)))
    rectangularity = float(np.clip(contour_area / candidate_area, 0.0, 1.0))
    area_score = float(np.clip(area_ratio / 0.30, 0.0, 1.0))
    if area_ratio > MAX_INTERNAL_CARD_AREA_RATIO:
        area_score *= float(
            np.clip((0.75 - area_ratio) / (0.75 - MAX_INTERNAL_CARD_AREA_RATIO), 0.0, 1.0)
        )

    polygon = np.round(ordered).astype(np.int32)
    interior_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillConvexPoly(interior_mask, polygon, 255)
    interior_pixels = int(np.count_nonzero(interior_mask))
    if interior_pixels == 0:
        return None
    interior_edges = float(
        np.count_nonzero((edge_union > 0) & (interior_mask > 0)) / interior_pixels
    )
    detail_score = float(np.clip(interior_edges / 0.08, 0.0, 1.0))

    border_mask = np.zeros(gray.shape, dtype=np.uint8)
    border_thickness = max(2, int(round(min(gray.shape[:2]) * 0.003)))
    cv2.polylines(
        border_mask,
        [polygon],
        True,
        255,
        thickness=border_thickness,
    )
    border_pixels = int(np.count_nonzero(border_mask))
    border_support = (
        float(np.count_nonzero((edge_union > 0) & (border_mask > 0)) / border_pixels)
        if border_pixels
        else 0.0
    )
    border_score = float(np.clip(border_support / 0.18, 0.0, 1.0))

    return float(
        0.45 * ratio_score
        + 0.15 * area_score
        + 0.15 * rectangularity
        + 0.15 * detail_score
        + 0.10 * border_score
    )



def _merged_projection_bounds(
    profile: np.ndarray,
    threshold: float,
    minimum_length: int,
    maximum_gap: int,
) -> tuple[int, int] | None:
    """Return the strongest group of dense edge-profile segments."""
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(profile > threshold):
        if active and start is None:
            start = index
        if start is not None and (not active or index == len(profile) - 1):
            end = index if active else index - 1
            if end - start + 1 >= minimum_length:
                segments.append((start, end))
            start = None
    if not segments:
        return None

    groups: list[list[tuple[int, int]]] = [[segments[0]]]
    for segment in segments[1:]:
        if segment[0] - groups[-1][-1][1] - 1 <= maximum_gap:
            groups[-1].append(segment)
        else:
            groups.append([segment])
    strongest = max(
        groups,
        key=lambda group: (
            sum(end - start + 1 for start, end in group),
            group[-1][1] - group[0][0],
        ),
    )
    return strongest[0][0], strongest[-1][1]




def _card_candidate_from_edge_projections(
    gray: np.ndarray,
) -> np.ndarray | None:
    """Infer a printed card box from dense text/edge bands on white paper."""
    edges = cv2.Canny(gray, 30, 100)
    height, width = edges.shape
    row_density = np.mean(edges > 0, axis=1).astype(np.float32)
    column_density = np.mean(edges > 0, axis=0).astype(np.float32)
    row_window = max(15, int(round(height * 0.02)))
    column_window = max(15, int(round(width * 0.03)))
    row_density = np.convolve(
        row_density,
        np.ones(row_window, dtype=np.float32) / row_window,
        mode="same",
    )
    column_density = np.convolve(
        column_density,
        np.ones(column_window, dtype=np.float32) / column_window,
        mode="same",
    )
    y_bounds = _merged_projection_bounds(
        row_density,
        threshold=0.02,
        minimum_length=max(10, int(round(height * 0.01))),
        maximum_gap=max(20, int(round(height * 0.12))),
    )
    x_bounds = _merged_projection_bounds(
        column_density,
        threshold=0.015,
        minimum_length=max(10, int(round(width * 0.01))),
        maximum_gap=max(20, int(round(width * 0.25))),
    )
    if x_bounds is None or y_bounds is None:
        return None

    x_padding = int(round(width * 0.025))
    y_padding = int(round(height * 0.02))
    x1 = max(0, x_bounds[0] - x_padding)
    x2 = min(width - 1, x_bounds[1] + x_padding)
    y1 = max(0, y_bounds[0] - y_padding)
    y2 = min(height - 1, y_bounds[1] + y_padding)
    if x2 <= x1 or y2 <= y1:
        return None

    # Edge projections mostly see printed text, not pale plastic/paper edges.
    # Expand the horizontal content box toward the physical card aspect ratio,
    # while retaining a small image-border guard against full-page crops.
    border_guard = max(3, int(round(min(width, height) * 0.01)))
    target_width = min(
        width - 2 * border_guard,
        max(x2 - x1, int(round((y2 - y1) * EXPECTED_CARD_RATIO))),
    )
    center_x = (x1 + x2) / 2.0
    x1 = int(round(center_x - target_width / 2.0))
    x2 = x1 + target_width
    if x1 < border_guard:
        x2 += border_guard - x1
        x1 = border_guard
    if x2 > width - 1 - border_guard:
        x1 -= x2 - (width - 1 - border_guard)
        x2 = width - 1 - border_guard
    return np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )




def _detect_card_from_multichannel_fallback(
    frame: np.ndarray,
) -> np.ndarray | None:
    """Find low-contrast printed cards using several color/edge views."""
    height, width = frame.shape[:2]
    longest_side = max(height, width)
    scale = min(1.0, CARD_DETECTION_FALLBACK_MAX_DIMENSION / longest_side)
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

    if detection_frame.ndim == 2:
        gray = detection_frame
        saturation = np.zeros_like(gray)
    else:
        gray = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)
        saturation = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2HSV)[:, :, 1]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred_gray = cv2.GaussianBlur(clahe, (5, 5), 0)
    blurred_saturation = cv2.GaussianBlur(saturation, (5, 5), 0)
    edge_masks = [
        cv2.Canny(blurred_gray, 30, 90),
        cv2.Canny(blurred_gray, 70, 180),
        cv2.Canny(blurred_saturation, 20, 80),
    ]
    adaptive_mask = cv2.adaptiveThreshold(
        blurred_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        7,
    )
    _, saturation_mask = cv2.threshold(
        blurred_saturation,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Printed photocopies can have no visible outer edge. Join their separated
    # text, MRZ, chip and pale color regions into one card-sized content block.
    content_mask = cv2.bitwise_or(adaptive_mask, saturation_mask)
    content_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            max(21, int(round(gray.shape[1] * 0.035))),
            max(21, int(round(gray.shape[0] * 0.025))),
        ),
    )
    content_mask = cv2.morphologyEx(
        content_mask,
        cv2.MORPH_CLOSE,
        content_kernel,
        iterations=2,
    )

    kernel_size = max(7, int(round(min(gray.shape[:2]) * 0.012)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_size, kernel_size),
    )
    candidate_masks = [
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel, iterations=2)
        for mask in [*edge_masks, adaptive_mask, saturation_mask, content_mask]
    ]
    edge_union = edge_masks[0].copy()
    for mask in edge_masks[1:]:
        edge_union = cv2.bitwise_or(edge_union, mask)

    frame_area = float(gray.shape[0] * gray.shape[1])
    best_points: np.ndarray | None = None
    best_score = 0.52
    for mask in candidate_masks:
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:100]:
            if cv2.contourArea(contour) < frame_area * 0.01:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            candidates: list[np.ndarray] = []
            if perimeter > 0:
                polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
                if len(polygon) == 4 and cv2.isContourConvex(polygon):
                    candidates.append(order_points(polygon))
            candidates.append(order_points(cv2.boxPoints(cv2.minAreaRect(contour))))

            for candidate in candidates:
                score = _score_multichannel_card_candidate(
                    candidate,
                    contour,
                    gray,
                    edge_union,
                    frame_area,
                )
                if score is not None and score > best_score:
                    best_score = score
                    best_points = candidate

    projection_candidate = (
        _card_candidate_from_edge_projections(cv2.GaussianBlur(gray, (5, 5), 0))
        if best_points is None
        else None
    )
    if projection_candidate is not None:
        projection_contour = np.round(projection_candidate).astype(np.int32)
        score = _score_multichannel_card_candidate(
            projection_candidate,
            projection_contour,
            gray,
            edge_union,
            frame_area,
        )
        if score is not None and score >= 0.45:
            best_points = projection_candidate

    return best_points / scale if best_points is not None else None
