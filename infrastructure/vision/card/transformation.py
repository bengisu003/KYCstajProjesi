"""Add crop padding and normalize detected cards by perspective warp."""

import cv2
import numpy as np

from core.config import (
    CARD_CROP_PADDING_MAX_PIXELS,
    CARD_CROP_PADDING_PIXELS,
    CARD_CROP_PADDING_RATIO,
    CARD_DETECTION_MODE,
    CARD_HEIGHT,
    CARD_WIDTH,
)
from infrastructure.vision.card.geometry import order_points


def _add_card_crop_padding(
    points: np.ndarray,
    frame_shape: tuple[int, ...],
) -> np.ndarray:
    """Expand detected corners by a small source-image margin."""
    ordered = order_points(points)
    padding_pixels = float(CARD_CROP_PADDING_PIXELS)
    if CARD_DETECTION_MODE.lower() == "improved":
        widths = [
            np.linalg.norm(ordered[1] - ordered[0]),
            np.linalg.norm(ordered[2] - ordered[3]),
        ]
        heights = [
            np.linalg.norm(ordered[3] - ordered[0]),
            np.linalg.norm(ordered[2] - ordered[1]),
        ]
        shorter_side = float(min(np.mean(widths), np.mean(heights)))
        padding_pixels = float(
            np.clip(
                shorter_side * CARD_CROP_PADDING_RATIO,
                CARD_CROP_PADDING_PIXELS,
                CARD_CROP_PADDING_MAX_PIXELS,
            )
        )
    if padding_pixels <= 0:
        return ordered
    center = np.mean(ordered, axis=0)
    vectors = ordered - center
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    expanded = ordered + padding_pixels * vectors / np.maximum(lengths, 1.0)
    frame_height, frame_width = frame_shape[:2]
    expanded[:, 0] = np.clip(expanded[:, 0], 0, max(frame_width - 1, 0))
    expanded[:, 1] = np.clip(expanded[:, 1], 0, max(frame_height - 1, 0))
    return expanded.astype(np.float32)




def warp_card(frame: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Warp a detected card and rotate portrait detections to landscape."""
    source_points = _add_card_crop_padding(points, frame.shape)

    average_width = (
        np.linalg.norm(source_points[1] - source_points[0])
        + np.linalg.norm(source_points[2] - source_points[3])
    ) / 2.0
    average_height = (
        np.linalg.norm(source_points[3] - source_points[0])
        + np.linalg.norm(source_points[2] - source_points[1])
    ) / 2.0

    if average_height > average_width:
        portrait_destination = np.array(
            [
                [0, 0],
                [CARD_HEIGHT - 1, 0],
                [CARD_HEIGHT - 1, CARD_WIDTH - 1],
                [0, CARD_WIDTH - 1],
            ],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(
            source_points,
            portrait_destination,
        )
        portrait_card = cv2.warpPerspective(
            frame,
            transform,
            (CARD_HEIGHT, CARD_WIDTH),
        )
        return cv2.rotate(portrait_card, cv2.ROTATE_90_CLOCKWISE)

    destination_points = np.array(
        [
            [0, 0],
            [CARD_WIDTH - 1, 0],
            [CARD_WIDTH - 1, CARD_HEIGHT - 1],
            [0, CARD_HEIGHT - 1],
        ],
        dtype=np.float32,
    )

    transform = cv2.getPerspectiveTransform(source_points, destination_points)
    return cv2.warpPerspective(frame, transform, (CARD_WIDTH, CARD_HEIGHT))
