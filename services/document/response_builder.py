"""Build public response payloads for document analysis workflows."""

from core.config import (
    DOCUMENT_SESSION_TTL_SECONDS,
    MODEL_VERSION,
)
from services.document.analysis_helpers import round_metrics
from services.document.side_detection import get_single_document_side_thresholds


SINGLE_DOCUMENT_CLASSIFICATION_WARNING = (
    "Card side and real/photocopy labels are heuristic image-risk predictions, "
    "not definitive authenticity decisions or calibrated probabilities."
)


def _optional_document_metrics(
    analysis: dict[str, object],
    key: str,
) -> dict[str, float] | None:
    """Return rounded numeric metrics when an optional analysis block exists."""
    metrics = analysis.get(key)
    return round_metrics(metrics) if isinstance(metrics, dict) else None


def build_document_classification_response(
    *,
    analysis: dict[str, object],
    decision: str,
    detected_side: str | None,
    front_side_score: float | None,
    document_score: float | None,
    classification_thresholds: dict[str, float] | None,
    reason_codes: list[str],
    saved_crop: dict[str, str] | None = None,
    document_session_id: str | None = None,
    source_filename: str | None = None,
) -> dict[str, object]:
    """Build the public response for the unified single-image workflow."""
    hologram_ready = document_session_id is not None
    return {
        "decision": decision,
        "detected_side": detected_side,
        "front_side_score": (
            round(front_side_score, 4) if front_side_score is not None else None
        ),
        "document_score": (
            round(document_score, 6) if document_score is not None else None
        ),
        "side_thresholds": get_single_document_side_thresholds(),
        "classification_thresholds": classification_thresholds,
        "reason_codes": reason_codes,
        "card_detected": bool(analysis["card_detected"]),
        "image_size": analysis["image_size"],
        "capture_quality": analysis.get("capture_quality"),
        "features": _optional_document_metrics(analysis, "features"),
        "normalized_components": _optional_document_metrics(
            analysis, "normalized_components"
        ),
        "saved_crop": saved_crop,
        "hologram_ready": hologram_ready,
        "hologram_session_expires_in_seconds": (
            DOCUMENT_SESSION_TTL_SECONDS if hologram_ready else None
        ),
        "document_session_id": document_session_id,
        "source_filename": source_filename,
        "model_version": MODEL_VERSION,
        "warning": SINGLE_DOCUMENT_CLASSIFICATION_WARNING,
    }
