"""Apply document and hologram rules when persisting analysis crops."""

import logging
from pathlib import Path
import threading
import time
from typing import Literal
from uuid import uuid4

import cv2
import numpy as np

from core.config import (
    CROP_CLEANUP_INTERVAL_SECONDS,
    CROP_OUTPUT_DIR,
    CROP_RETENTION_DAYS,
)
from core.exceptions import CropSaveError
from infrastructure.storage.jpeg_writer import write_jpeg


DocumentLabel = Literal["real", "photocopy"]
DocumentSide = Literal["front", "back"]

logger = logging.getLogger(__name__)
_crop_cleanup_lock = threading.Lock()
_next_crop_cleanup_at = 0.0


def cleanup_expired_crop_files(
    *,
    force: bool = False,
    current_time: float | None = None,
) -> int:
    """Delete expired JPEG crops under the configured crop root only."""
    global _next_crop_cleanup_at
    monotonic_now = time.monotonic()
    with _crop_cleanup_lock:
        if not force and monotonic_now < _next_crop_cleanup_at:
            return 0
        _next_crop_cleanup_at = monotonic_now + CROP_CLEANUP_INTERVAL_SECONDS

        crop_root = CROP_OUTPUT_DIR.resolve()
        if not crop_root.exists():
            return 0
        cutoff = (
            time.time() if current_time is None else current_time
        ) - CROP_RETENTION_DAYS * 24 * 60 * 60
        removed_count = 0
        try:
            crop_files = list(crop_root.rglob("*.jpg"))
        except OSError:
            logger.warning("Expired crop discovery failed.")
            return 0

        for crop_file in crop_files:
            try:
                resolved_file = crop_file.resolve(strict=True)
                if not resolved_file.is_relative_to(crop_root):
                    continue
                if resolved_file.is_file() and resolved_file.stat().st_mtime < cutoff:
                    resolved_file.unlink()
                    removed_count += 1
            except OSError:
                logger.warning("An expired crop could not be removed.")
        if removed_count:
            logger.info("Removed %d expired analysis crop(s).", removed_count)
        return removed_count

# UUID sample_id oluşturur; real/photocopy ve front/back
# yolunda JPEG yazar
def _save_card_crop(
    card_crop: np.ndarray,
    label: DocumentLabel,
    side: DocumentSide,
) -> tuple[dict[str, str], Path]:
    """Write one labeled card crop and return public metadata and target path."""
    if not isinstance(card_crop, np.ndarray):
        raise CropSaveError("The normalized document-card crop is unavailable.")
    if label not in {"real", "photocopy"}:
        raise ValueError("Document label must be 'real' or 'photocopy'.")
    if side not in {"front", "back"}:
        raise ValueError("Document side must be 'front' or 'back'.")

    cleanup_expired_crop_files()
    sample_id = uuid4().hex
    relative_path = Path(label) / side / f"{sample_id}.jpg"
    target_path = CROP_OUTPUT_DIR / relative_path
    try:
        write_jpeg(target_path, card_crop)
    except (OSError, cv2.error, CropSaveError) as error:
        target_path.unlink(missing_ok=True)
        raise CropSaveError("The document-card crop could not be saved.") from error

    return {
        "sample_id": sample_id,
        "card_crop": str(Path("cropped_images") / relative_path),
    }, target_path


def save_document_card_crop(
    card_crop: np.ndarray,
    label: DocumentLabel,
    side: DocumentSide,
) -> dict[str, str]:
    """Save one classified front/back card crop under its predicted label."""
    saved_crop, _target_path = _save_card_crop(card_crop, label, side)
    return saved_crop


def save_real_hologram_crop(
    hologram_crop: np.ndarray,
    sample_id: str,
) -> dict[str, str]:
    """Save the hologram ROI belonging to an authorized real-card sample."""
    cleanup_expired_crop_files()
    relative_path = Path("real") / "hologram_crops" / f"{sample_id}.jpg"
    target_path = CROP_OUTPUT_DIR / relative_path
    try:
        write_jpeg(target_path, hologram_crop)
    except (OSError, cv2.error, CropSaveError) as error:
        target_path.unlink(missing_ok=True)
        raise CropSaveError("The real-card hologram crop could not be saved.") from error
    return {
        "sample_id": sample_id,
        "hologram_crop": str(Path("cropped_images") / relative_path),
    }
