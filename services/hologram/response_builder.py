"""Build the public hologram decision response."""

from core.config import (
    MODEL_VERSION,
    STATIC_HOLOGRAM_ABSENT_MAX_SCORE,
    STATIC_HOLOGRAM_PRESENT_MIN_SCORE,
)
from infrastructure.vision.card import to_finite_float


def _rounded_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Round finite metrics before including them in an API response."""
    return {key: round(to_finite_float(value), 4) for key, value in metrics.items()}


def build_hologram_analysis_response(
    analysis: dict[str, object],
    document_session_id: str,
    decision: str,
    reason_codes: list[str],
    score: float | None = None,
    saved_hologram_crop: dict[str, str] | None = None,
    source_frame: str | None = None,
    source_filename: str | None = None,
) -> dict[str, object]:
    """Build the public response for one authorized hologram analysis."""
    if not analysis["valid"]:
        return {
            "decision": decision,
            "hologram_score": None,
            "reason_codes": reason_codes,
            "card_detected": bool(analysis["card_detected"]),
            "image_size": analysis["image_size"],
            "capture_quality": analysis.get("capture_quality"),
            "document_session_id": document_session_id,
            "source_frame": source_frame,
            "source_filename": source_filename,
            "model_version": MODEL_VERSION,
        }

    if score is None:
        raise ValueError("A valid hologram analysis requires a score.")

    return {
        "decision": decision,
        "hologram_score": round(score, 4),
        "thresholds": {
            "hologram_absent_max_score": STATIC_HOLOGRAM_ABSENT_MAX_SCORE,
            "hologram_present_min_score": STATIC_HOLOGRAM_PRESENT_MIN_SCORE,
        },
        "reason_codes": reason_codes,
        "card_detected": True,
        "image_size": analysis["image_size"],
        "capture_quality": analysis["capture_quality"],
        "features": _rounded_metrics(analysis["features"]),
        "normalized_components": _rounded_metrics(
            analysis["normalized_components"]
        ),
        "saved_hologram_crop": saved_hologram_crop,
        "document_session_id": document_session_id,
        "source_frame": source_frame,
        "source_filename": source_filename,
        "model_version": MODEL_VERSION,
        "warning": (
            "Single-frame analysis is only a risk signal for static hologram presence."
        ),
    }
