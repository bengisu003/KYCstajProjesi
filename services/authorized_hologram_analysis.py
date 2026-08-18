"""Run the authorized static hologram analysis workflow."""

from core.config import (
    STATIC_HOLOGRAM_ABSENT_MAX_SCORE,
    STATIC_HOLOGRAM_PRESENT_MIN_SCORE,
)
from infrastructure.storage.crop_storage import save_real_hologram_crop
from infrastructure.session.hologram_authorization import consume_real_candidate
from services.hologram.frame_analysis import analyze_hologram_card
from services.hologram.response_builder import build_hologram_analysis_response


def _classify_hologram_score(score: float) -> tuple[str, list[str]]:
    """Classify one valid static-hologram score with configured thresholds."""
    if score >= STATIC_HOLOGRAM_PRESENT_MIN_SCORE:
        return "hologram_detected", ["STATIC_HOLOGRAM_EVIDENCE_DETECTED"]
    if score <= STATIC_HOLOGRAM_ABSENT_MAX_SCORE:
        return "hologram_not_detected", ["LOW_STATIC_HOLOGRAM_EVIDENCE"]
    return "retake_required", ["INCONCLUSIVE_STATIC_HOLOGRAM_SIGNAL"]

# Yetkili gerçek ön yüzü tüketme, hologram analizi ve response
# üretimini sıraya koya
def analyze_authorized_hologram(document_session_id: str) -> dict[str, object]:
    """Consume and analyze the real front-card crop cached by document session."""
    (
        card_crop,
        image_size,
        authorized_sample_id,
        source_frame,
        source_filename,
    ) = consume_real_candidate(document_session_id)

    analysis = analyze_hologram_card(card_crop, image_size)
    if not analysis["valid"]:
        return build_hologram_analysis_response(
            analysis,
            document_session_id,
            "retake_required",
            list(analysis["reason_codes"]),
            source_frame=source_frame,
            source_filename=source_filename,
        )

    score = float(analysis["static_hologram_score"])
    saved_hologram_crop = save_real_hologram_crop(
        analysis["hologram_crop"],
        authorized_sample_id,
    )
    decision, reason_codes = _classify_hologram_score(score)
    return build_hologram_analysis_response(
        analysis,
        document_session_id,
        decision,
        reason_codes,
        score=score,
        saved_hologram_crop=saved_hologram_crop,
        source_frame=source_frame,
        source_filename=source_filename,
    )
