"""Detect whether one normalized identity card shows its front or back side."""

import math
from typing import Literal

import cv2
import numpy as np

from core.config import (
    MAX_BACK_SIDE_ANCHOR_SCORE,
    MIN_FRONT_SIDE_ANCHOR_SCORE,
)
from services.document.back_security_analysis import (
    calculate_back_document_score,
    extract_back_document_features,
)
from services.document.side_classifier import (
    EDGE_GRID_COLUMNS,
    EDGE_GRID_ROWS,
    SIDE_FEATURE_NAMES,
    score_front_side_features,
)


DetectedDocumentSide = Literal["front", "back"]


def get_single_document_side_thresholds() -> dict[str, float]:
    """Return the absolute front/back limits used for one normalized card."""
    return {
        "front_min_score": MIN_FRONT_SIDE_ANCHOR_SCORE,
        "back_max_score": MAX_BACK_SIDE_ANCHOR_SCORE,
    }


def classify_single_document_side(
    front_score: float,
) -> tuple[DetectedDocumentSide | None, list[str]]:
    """Classify one front-evidence score without relying on a four-card ranking."""
    converted_score = float(front_score)
    if not math.isfinite(converted_score) or not 0.0 <= converted_score <= 1.0:
        return None, ["DOCUMENT_SIDE_FEATURES_UNAVAILABLE"]
    if converted_score >= MIN_FRONT_SIDE_ANCHOR_SCORE:
        return "front", ["FRONT_SIDE_DETECTED"]
    if converted_score <= MAX_BACK_SIDE_ANCHOR_SCORE:
        return "back", ["BACK_SIDE_DETECTED"]
    return None, ["AMBIGUOUS_DOCUMENT_SIDE"]


def extract_front_side_features(
    analysis: dict[str, object],
) -> dict[str, float] | None:
    """Extract non-identity layout evidence for front/back classification."""
    card = analysis.get("card_crop")
    features = analysis.get("features")
    if not isinstance(card, np.ndarray) or not isinstance(features, dict):
        return None

    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    lower = gray[
        int(0.63 * height) : int(0.97 * height),
        int(0.04 * width) : int(0.96 * width),
    ]
    lower_edge_density = float(np.mean(cv2.Canny(lower, 60, 160) > 0))
    flag_red_ratio = float(features.get("flag_red_ratio", 0.0))
    portrait = gray[
        int(0.32 * height) : int(0.90 * height),
        int(0.05 * width) : int(0.38 * width),
    ]
    _threshold, portrait_mask = cv2.threshold(
        portrait, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    _count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        portrait_mask, connectivity=8
    )
    portrait_component_ratio = float(
        max(stats[1:, cv2.CC_STAT_AREA], default=0) / portrait.size
    )
    portrait_dark_ratio = float(np.mean(portrait_mask > 0))
    back_features = extract_back_document_features(card)
    back_score, _back_components = calculate_back_document_score(back_features)
    side_features = {
        "flag_red_ratio": flag_red_ratio,
        "portrait_component_ratio": portrait_component_ratio,
        "portrait_dark_ratio": portrait_dark_ratio,
        "lower_edge_density": lower_edge_density,
        "front_document_score": float(analysis.get("document_score", 0.0)),
        "back_document_score": back_score,
        "mrz_contrast": float(back_features["mrz_contrast"]),
        "mrz_detail_variance": float(back_features["mrz_detail_variance"]),
        "edge_density": float(back_features["edge_density"]),
    }
    card_edges = cv2.Canny(gray, 60, 160) > 0
    edge_grid_values = (
        float(np.mean(cell))
        for row_band in np.array_split(card_edges, EDGE_GRID_ROWS, axis=0)
        for cell in np.array_split(row_band, EDGE_GRID_COLUMNS, axis=1)
    )
    side_features.update(
        dict(zip(SIDE_FEATURE_NAMES[9:], edge_grid_values, strict=True))
    )
    return side_features


def calculate_front_side_score(
    analysis: dict[str, object],
) -> tuple[float, dict[str, float]]:
    """Score front-side layout evidence without using identity text or data."""
    side_features = extract_front_side_features(analysis)
    if side_features is None:
        return -1.0, {"flag_red_ratio": 0.0, "lower_edge_density": 0.0}

    score = score_front_side_features(side_features)
    return score, {
        "flag_red_ratio": side_features["flag_red_ratio"],
        "lower_edge_density": side_features["lower_edge_density"],
        "portrait_component_ratio": side_features["portrait_component_ratio"],
        "mrz_contrast": side_features["mrz_contrast"],
    }
