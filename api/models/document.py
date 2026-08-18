"""Pydantic response models for document-analysis endpoints."""

from typing import Literal

from pydantic import BaseModel

from api.models.common import CaptureQuality, ImageSize


class SavedCardCrop(BaseModel):
    sample_id: str
    card_crop: str


class DocumentCheckResponse(BaseModel):
    """Side and real/photocopy classification for one document image."""

    decision: Literal[
        "real_candidate", "photocopy_suspected", "retake_required"
    ]
    detected_side: Literal["front", "back"] | None
    front_side_score: float | None
    document_score: float | None
    side_thresholds: dict[str, float]
    classification_thresholds: dict[str, float] | None
    reason_codes: list[str]
    card_detected: bool
    image_size: ImageSize
    capture_quality: CaptureQuality | None = None
    features: dict[str, float] | None = None
    normalized_components: dict[str, float] | None = None
    saved_crop: SavedCardCrop | None = None
    hologram_ready: bool = False
    hologram_session_expires_in_seconds: int | None = None
    document_session_id: str | None = None
    source_filename: str | None = None
    model_version: str
    warning: str
