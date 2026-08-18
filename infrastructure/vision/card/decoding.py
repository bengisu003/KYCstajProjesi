"""Validate and decode bounded uploaded image bytes."""

import cv2
import numpy as np

from core.config import MAX_FRAME_PIXELS, MAX_FRAME_SIZE_MB


class InvalidFrameError(ValueError):
    """Raised when uploaded bytes are not a valid bounded image."""


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_START_OF_FRAME_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


def _read_png_size(frame_bytes: bytes) -> tuple[int, int] | None:
    """Return PNG dimensions from its mandatory IHDR chunk."""
    if not frame_bytes.startswith(_PNG_SIGNATURE):
        return None
    if (
        len(frame_bytes) < 24
        or frame_bytes[8:12] != b"\x00\x00\x00\x0d"
        or frame_bytes[12:16] != b"IHDR"
    ):
        raise InvalidFrameError("The uploaded PNG header is invalid.")

    width = int.from_bytes(frame_bytes[16:20], "big")
    height = int.from_bytes(frame_bytes[20:24], "big")
    if width <= 0 or height <= 0:
        raise InvalidFrameError("The uploaded image dimensions are invalid.")
    return width, height


def _read_jpeg_size(frame_bytes: bytes) -> tuple[int, int] | None:
    """Return JPEG dimensions by walking marker segments before scan data."""
    if not frame_bytes.startswith(b"\xff\xd8"):
        return None

    offset = 2
    byte_count = len(frame_bytes)
    while offset < byte_count:
        while offset < byte_count and frame_bytes[offset] != 0xFF:
            offset += 1
        while offset < byte_count and frame_bytes[offset] == 0xFF:
            offset += 1
        if offset >= byte_count:
            break

        marker = frame_bytes[offset]
        offset += 1
        if marker in {0xD9, 0xDA}:
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:
            continue
        if offset + 2 > byte_count:
            break

        segment_length = int.from_bytes(frame_bytes[offset : offset + 2], "big")
        if segment_length < 2 or offset + segment_length > byte_count:
            raise InvalidFrameError("The uploaded JPEG header is invalid.")
        if marker in _JPEG_START_OF_FRAME_MARKERS:
            if segment_length < 7:
                raise InvalidFrameError("The uploaded JPEG header is invalid.")
            height = int.from_bytes(frame_bytes[offset + 3 : offset + 5], "big")
            width = int.from_bytes(frame_bytes[offset + 5 : offset + 7], "big")
            if width <= 0 or height <= 0:
                raise InvalidFrameError("The uploaded image dimensions are invalid.")
            return width, height
        offset += segment_length

    raise InvalidFrameError("The uploaded JPEG dimensions could not be read.")


def _read_encoded_image_size(frame_bytes: bytes) -> tuple[int, int]:
    """Return JPEG/PNG dimensions without allocating a decoded image."""
    png_size = _read_png_size(frame_bytes)
    if png_size is not None:
        return png_size
    jpeg_size = _read_jpeg_size(frame_bytes)
    if jpeg_size is not None:
        return jpeg_size
    raise InvalidFrameError("The uploaded frame is not a supported JPEG or PNG image.")


def decode_frame_bytes(frame_bytes: bytes) -> np.ndarray:
    """Decode bounded JPEG/PNG bytes without writing them to disk."""
    if not isinstance(frame_bytes, bytes) or not frame_bytes:
        raise InvalidFrameError("The uploaded frame cannot be empty.")

    maximum_bytes = MAX_FRAME_SIZE_MB * 1024 * 1024
    if len(frame_bytes) > maximum_bytes:
        raise InvalidFrameError(
            f"An uploaded frame can be at most {MAX_FRAME_SIZE_MB} MB."
        )

    encoded_width, encoded_height = _read_encoded_image_size(frame_bytes)
    if encoded_width * encoded_height > MAX_FRAME_PIXELS:
        raise InvalidFrameError("The frame exceeds the pixel limit.")

    image_buffer = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise InvalidFrameError("The uploaded frame is not a valid image.")

    height, width = frame.shape[:2]
    if height * width > MAX_FRAME_PIXELS:
        raise InvalidFrameError("The frame exceeds the pixel limit.")
    return frame
