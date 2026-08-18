"""Tests for bounded image decoding and shared numeric normalization."""

import unittest

from core.config import MAX_FRAME_PIXELS
from infrastructure.vision.card import (
    InvalidFrameError,
    decode_frame_bytes,
    normalize_feature,
)


class CardDecodingTests(unittest.TestCase):
    def test_empty_frame_is_rejected(self) -> None:
        with self.assertRaisesRegex(InvalidFrameError, "cannot be empty"):
            decode_frame_bytes(b"")

    def test_unsupported_image_bytes_are_rejected(self) -> None:
        with self.assertRaisesRegex(InvalidFrameError, "supported JPEG or PNG"):
            decode_frame_bytes(b"not-an-image")

    def test_png_dimensions_are_checked_before_decoding(self) -> None:
        oversized_png_header = (
            b"\x89PNG\r\n\x1a\n"
            + b"\x00\x00\x00\x0dIHDR"
            + (MAX_FRAME_PIXELS + 1).to_bytes(4, "big")
            + (1).to_bytes(4, "big")
        )
        with self.assertRaisesRegex(InvalidFrameError, "pixel limit"):
            decode_frame_bytes(oversized_png_header)

    def test_feature_normalization_is_bounded(self) -> None:
        self.assertEqual(normalize_feature(-1.0, 0.0, 10.0), 0.0)
        self.assertEqual(normalize_feature(5.0, 0.0, 10.0), 0.5)
        self.assertEqual(normalize_feature(11.0, 0.0, 10.0), 1.0)


if __name__ == "__main__":
    unittest.main()
