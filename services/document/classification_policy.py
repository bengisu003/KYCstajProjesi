"""Apply absolute single-image real/photocopy decision policies."""

import math
from typing import Literal

from core.config import (
    BACK_CLASSIFIER_PHOTOCOPY_MAX_SCORE,
    BACK_CLASSIFIER_REAL_MIN_SCORE,
    FRONT_CLASSIFIER_PHOTOCOPY_MAX_SCORE,
    FRONT_CLASSIFIER_REAL_MIN_SCORE,
)


DocumentSide = Literal["front", "back"]
DocumentDecision = Literal[
    "real_candidate", "photocopy_suspected", "retake_required"
]


def get_document_classification_thresholds(
    side: DocumentSide,
) -> dict[str, float]:
    """Return absolute single-image decision thresholds for one card side."""
    if side == "front":
        return {
            "photocopy_max_score": FRONT_CLASSIFIER_PHOTOCOPY_MAX_SCORE,
            "real_min_score": FRONT_CLASSIFIER_REAL_MIN_SCORE,
        }
    if side == "back":
        return {
            "photocopy_max_score": BACK_CLASSIFIER_PHOTOCOPY_MAX_SCORE,
            "real_min_score": BACK_CLASSIFIER_REAL_MIN_SCORE,
        }
    raise ValueError("Document side must be 'front' or 'back'.")


def classify_document_score(
    side: DocumentSide,
    score: float,
) -> tuple[DocumentDecision, list[str]]:
    """Classify one finite, normalized document score using side-specific limits."""
    converted_score = float(score)
    if not math.isfinite(converted_score) or not 0.0 <= converted_score <= 1.0:
        raise ValueError("Document score must be a finite value between 0 and 1.")

    thresholds = get_document_classification_thresholds(side)
    side_prefix = side.upper()
    if converted_score >= thresholds["real_min_score"]:
        return "real_candidate", [f"{side_prefix}_REAL_EVIDENCE_ABOVE_THRESHOLD"]
    if converted_score <= thresholds["photocopy_max_score"]:
        return "photocopy_suspected", [
            f"{side_prefix}_PHOTOCOPY_EVIDENCE_ABOVE_THRESHOLD"
        ]
    return "retake_required", [
        f"INCONCLUSIVE_{side_prefix}_DOCUMENT_CLASSIFICATION"
    ]
