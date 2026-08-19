"""Quality-check and analyze an already normalized real-card candidate."""

import cv2
import numpy as np

from core.config import MAX_BRIGHT_PIXEL_RATIO, MAX_DARK_PIXEL_RATIO, MIN_BLUR_SCORE
from infrastructure.vision.card import calculate_card_quality
from services.hologram.feature_analysis import (
    calculate_static_hologram_score,
    extract_static_hologram_features,
    extract_static_hologram_roi,
)


def analyze_hologram_card(
    warped_card: np.ndarray,
    image_size: dict[str, int],
) -> dict[str, object]:
    """Analyze a perspective-corrected card authorized by a document session."""
    try:
        roi = extract_static_hologram_roi(warped_card)
    except (cv2.error, ValueError):
        return {
            "valid": False,
            "card_detected": False,
            "image_size": image_size,
            "reason_codes": ["CARD_WARP_FAILED"],
        }

    blur_score, dark_ratio, bright_ratio = calculate_card_quality(warped_card)
    reason_codes: list[str] = []
    if blur_score < MIN_BLUR_SCORE:
        reason_codes.append("IMAGE_TOO_BLURRY")
    if dark_ratio > MAX_DARK_PIXEL_RATIO:
        reason_codes.append("IMAGE_TOO_DARK")
    if bright_ratio > MAX_BRIGHT_PIXEL_RATIO:
        reason_codes.append("IMAGE_OVEREXPOSED")

    features = extract_static_hologram_features(roi)
    score, components = calculate_static_hologram_score(features)
    return {
        "valid": not reason_codes,
        "card_detected": True,
        "image_size": image_size,
        "capture_quality": {
            "blur_score": round(blur_score, 4),
            "dark_ratio": round(dark_ratio, 4),
            "bright_ratio": round(bright_ratio, 4),
        },
        "reason_codes": reason_codes,
        "static_hologram_score": score,
        "features": features,
        "normalized_components": components,
        "card_crop": warped_card,
        "hologram_crop": roi,
    }
