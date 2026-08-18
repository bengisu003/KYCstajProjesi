"""Shared formatting and relative-ROI helpers for document analysis."""

import numpy as np

from infrastructure.vision.card import to_finite_float


def round_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Round finite metrics before including them in an API response."""
    return {key: round(to_finite_float(value), 4) for key, value in metrics.items()}


def extract_relative_roi(
    image: np.ndarray,
    x1_ratio: float,
    y1_ratio: float,
    x2_ratio: float,
    y2_ratio: float,
) -> np.ndarray:
    """Return a non-empty ROI from relative normalized-card coordinates."""
    height, width = image.shape[:2]
    x1 = int(round(x1_ratio * width))
    y1 = int(round(y1_ratio * height))
    x2 = int(round(x2_ratio * width))
    y2 = int(round(y2_ratio * height))
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        raise ValueError("Configured front-side color region is empty.")
    return roi
