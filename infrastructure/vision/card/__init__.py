"""Public card-processing API assembled from focused modules."""

from infrastructure.vision.card.decoding import InvalidFrameError, decode_frame_bytes
from infrastructure.vision.card.detection import detect_card
from infrastructure.vision.card.geometry import order_points
from infrastructure.vision.card.quality import (
    calculate_card_quality,
    normalize_feature,
    to_finite_float,
)
from infrastructure.vision.card.transformation import warp_card


__all__ = [
    "InvalidFrameError",
    "calculate_card_quality",
    "decode_frame_bytes",
    "detect_card",
    "normalize_feature",
    "order_points",
    "to_finite_float",
    "warp_card",
]
