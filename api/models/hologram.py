"""Pydantic response models for the hologram-analysis endpoint."""

from typing import Literal

from pydantic import BaseModel

from api.models.common import CaptureQuality, ImageSize


class SavedHologramCrop(BaseModel):
    sample_id: str
    hologram_crop: str


class HologramResponse(BaseModel):
    decision: Literal[
        "hologram_detected", "hologram_not_detected", "retake_required"
    ]
    hologram_score: float | None
    thresholds: dict[str, float] | None = None
    reason_codes: list[str]
    card_detected: bool
    image_size: ImageSize
    capture_quality: CaptureQuality | None = None
    features: dict[str, float] | None = None
    normalized_components: dict[str, float] | None = None
    saved_hologram_crop: SavedHologramCrop | None = None
    document_session_id: str | None = None
    source_frame: str | None = None
    source_filename: str | None = None
    model_version: str
    warning: str | None = None
