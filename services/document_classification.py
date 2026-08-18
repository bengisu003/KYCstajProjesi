"""Run the unified single-image document classification workflow."""

import numpy as np

from infrastructure.session.hologram_authorization import register_real_candidate
from infrastructure.storage.crop_storage import save_document_card_crop
from services.document.classification_policy import (
    classify_document_score,
    get_document_classification_thresholds,
)
from services.document.frame_analysis import (
    analyze_prepared_document_frame,
    prepare_document_frame,
)
from services.document.front_classifier import calculate_front_classifier_score
from services.document.response_builder import (
    build_document_classification_response,
)
from services.document.side_detection import (
    calculate_front_side_score,
    classify_single_document_side,
)


def classify_document(
    frame_bytes: bytes,
    source_filename: str | None = None,
) -> dict[str, object]:
    """Classify one card image by side, then by real/photocopy evidence."""
    prepared = prepare_document_frame(frame_bytes)
    if not prepared["valid"]:
        return build_document_classification_response(
            analysis=prepared,
            decision="retake_required",
            detected_side=None,
            front_side_score=None,
            document_score=None,
            classification_thresholds=None,
            reason_codes=list(prepared["reason_codes"]),
            source_filename=source_filename,
        )

    front_analysis = analyze_prepared_document_frame(prepared, "front")
    front_side_score, _side_evidence = calculate_front_side_score(front_analysis)
    detected_side, side_reason_codes = classify_single_document_side(
        front_side_score
    )
    if detected_side is None:
        return build_document_classification_response(
            analysis=front_analysis,
            decision="retake_required",
            detected_side=None,
            front_side_score=front_side_score,
            document_score=None,
            classification_thresholds=None,
            reason_codes=side_reason_codes,
            source_filename=source_filename,
        )

    if detected_side == "front":
        analysis = front_analysis
        components = analysis.get("normalized_components")
        if not isinstance(components, dict):
            return build_document_classification_response(
                analysis=analysis,
                decision="retake_required",
                detected_side=detected_side,
                front_side_score=front_side_score,
                document_score=None,
                classification_thresholds=(
                    get_document_classification_thresholds(detected_side)
                ),
                reason_codes=[*side_reason_codes, "FRONT_FEATURES_UNAVAILABLE"],
                source_filename=source_filename,
            )
        document_score = calculate_front_classifier_score(components)
    else:
        analysis = analyze_prepared_document_frame(prepared, "back")
        document_score = float(analysis["document_score"])

    classification_thresholds = get_document_classification_thresholds(
        detected_side
    )
    decision, classification_reason_codes = classify_document_score(
        detected_side,
        document_score,
    )

    saved_crop: dict[str, str] | None = None
    document_session_id: str | None = None
    if decision in {"real_candidate", "photocopy_suspected"}:
        card_crop = analysis.get("card_crop")
        if not isinstance(card_crop, np.ndarray):
            return build_document_classification_response(
                analysis=analysis,
                decision="retake_required",
                detected_side=detected_side,
                front_side_score=front_side_score,
                document_score=document_score,
                classification_thresholds=classification_thresholds,
                reason_codes=[*side_reason_codes, "CARD_CROP_UNAVAILABLE"],
                source_filename=source_filename,
            )
        label = "real" if decision == "real_candidate" else "photocopy"
        saved_crop = save_document_card_crop(card_crop, label, detected_side)
        if detected_side == "front" and decision == "real_candidate":
            document_session_id = register_real_candidate(
                saved_crop["sample_id"],
                card_crop,
                analysis["image_size"],
                "frame",
                source_filename,
            )

    return build_document_classification_response(
        analysis=analysis,
        decision=decision,
        detected_side=detected_side,
        front_side_score=front_side_score,
        document_score=document_score,
        classification_thresholds=classification_thresholds,
        reason_codes=[*side_reason_codes, *classification_reason_codes],
        saved_crop=saved_crop,
        document_session_id=document_session_id,
        source_filename=source_filename,
    )
