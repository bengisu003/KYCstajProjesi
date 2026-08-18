"""Response models shared by document and hologram endpoints."""

from pydantic import BaseModel


class ImageSize(BaseModel):
    width: int
    height: int


class CaptureQuality(BaseModel):
    blur_score: float
    dark_ratio: float
    bright_ratio: float
