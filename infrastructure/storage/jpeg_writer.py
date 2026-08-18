"""Encode OpenCV images and write them as JPEG files."""

from pathlib import Path

import cv2
import numpy as np

from core.exceptions import CropSaveError

# klasörü hazırlar, görüntüyü JPEG’e çevirir ve diske yazar. 
# Dosyanın real, photocopy, front veya back klasörlerinden hangisine kaydedileceğine ise bu fonksiyon karar vermez; 
# bu kararı crop_storage.py verir.
def write_jpeg(path: Path, image: np.ndarray) -> None:
    """Encode and write one JPEG image, including on Windows Unicode paths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded, jpeg_buffer = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    if not encoded:
        raise CropSaveError("An analysis crop could not be saved.")
    path.write_bytes(jpeg_buffer.tobytes())
