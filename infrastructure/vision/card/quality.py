"""Calculate card quality and normalized numeric features."""

import cv2
import numpy as np


def calculate_card_quality(warped_card: np.ndarray) -> tuple[float, float, float]:
    """Return blur, dark-pixel, and bright-pixel metrics for one card crop."""
    gray = cv2.cvtColor(warped_card, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    dark_ratio = float(np.mean(gray < 25))
    bright_ratio = float(np.mean(gray > 245))
    return blur_score, dark_ratio, bright_ratio


def to_finite_float(value: float, default: float = 0.0) -> float:
    """Convert numeric values to finite built-in floats for JSON responses."""
    converted = float(value)
    return converted if np.isfinite(converted) else default


def normalize_feature(value: float, low: float, high: float) -> float:
    """Normalize a feature to the inclusive 0-1 interval."""
    if high <= low:
        raise ValueError("Özellik üst sınırı alt sınırdan büyük olmalıdır.")
    converted = float(value)
    if not np.isfinite(converted):
        raise ValueError("Özellik değeri sonlu bir sayı olmalıdır.")
    return float(np.clip((converted - low) / (high - low), 0.0, 1.0))
