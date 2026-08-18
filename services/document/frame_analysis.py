"""Decode, detect, normalize, quality-check, and analyze one document frame."""

import cv2
import numpy as np

from core.config import (
    MAX_BRIGHT_PIXEL_RATIO,
    MAX_DARK_PIXEL_RATIO,
    MIN_BLUR_SCORE,
)
from infrastructure.vision.card import (
    calculate_card_quality,
    decode_frame_bytes,
    detect_card,
    warp_card,
)
from services.document.back_security_analysis import (
    calculate_back_document_score,
    extract_back_document_features,
)
from services.document.front_color_analysis import (
    calculate_front_color_score,
    extract_front_color_features,
)


def prepare_document_frame(frame_bytes: bytes) -> dict[str, object]:
    """Decode and normalize one uploaded frame independently of document side."""
    frame = decode_frame_bytes(frame_bytes)
    height, width = frame.shape[:2]
    card_points = detect_card(frame)
    if card_points is None:
        return {
            "valid": False,
            "card_detected": False,
            "image_size": {"width": width, "height": height},
            "reason_codes": ["CARD_NOT_DETECTED"],
        }

    try:
        warped_card = warp_card(frame, card_points)
    except (cv2.error, ValueError):
        return {
            "valid": False,
            "card_detected": False,
            "image_size": {"width": width, "height": height},
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

    return {
        "valid": not reason_codes,
        "card_detected": True,
        "image_size": {"width": width, "height": height},
        "capture_quality": {
            "blur_score": round(blur_score, 4),
            "dark_ratio": round(dark_ratio, 4),
            "bright_ratio": round(bright_ratio, 4),
        },
        "reason_codes": reason_codes,
        "card_crop": warped_card,
    }


def analyze_prepared_document_frame(
    prepared_frame: dict[str, object],
    side: str = "front",
) -> dict[str, object]:
    """Apply side-specific feature extraction to an already normalized frame."""
    if side not in {"front", "back"}:
        raise ValueError("Document side must be 'front' or 'back'.")

    analysis = dict(prepared_frame)
    warped_card = analysis.get("card_crop")
    if warped_card is not None and not isinstance(warped_card, np.ndarray):
        raise ValueError("Prepared document frame contains an invalid card crop.")
    if warped_card is None:
        return analysis

    if side == "back":
        features = extract_back_document_features(warped_card)
        score, components = calculate_back_document_score(features)
    else:
        features = extract_front_color_features(warped_card)
        score, components = calculate_front_color_score(features)
    analysis.update(
        {
            "document_score": score,
            "features": features,
            "normalized_components": components,
        }
    )
    return analysis
