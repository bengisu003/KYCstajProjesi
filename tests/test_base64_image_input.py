"""Tests for bounded Base64 JPEG/PNG API input decoding."""

import base64
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from api.image_uploads import decode_base64_image


class Base64ImageInputTests(unittest.TestCase):
    def test_raw_jpeg_base64_is_decoded(self) -> None:
        content = b"\xff\xd8\xffjpeg-content"
        encoded = base64.b64encode(content).decode("ascii")

        self.assertEqual(decode_base64_image(encoded, "image/jpeg"), content)

    def test_matching_png_data_url_is_decoded(self) -> None:
        content = b"\x89PNG\r\n\x1a\npng-content"
        encoded = base64.b64encode(content).decode("ascii")

        self.assertEqual(
            decode_base64_image(
                f"data:image/png;base64,{encoded}",
                "image/png",
            ),
            content,
        )

    def test_invalid_base64_is_rejected(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            decode_base64_image("not-valid-base64!", "image/jpeg")

        self.assertEqual(raised.exception.status_code, 422)

    def test_content_must_match_declared_media_type(self) -> None:
        png_content = b"\x89PNG\r\n\x1a\npng-content"
        encoded = base64.b64encode(png_content).decode("ascii")

        with self.assertRaises(HTTPException) as raised:
            decode_base64_image(encoded, "image/jpeg")

        self.assertEqual(raised.exception.status_code, 422)

    def test_unsupported_media_type_is_rejected(self) -> None:
        encoded = base64.b64encode(b"gif-content").decode("ascii")

        with self.assertRaises(HTTPException) as raised:
            decode_base64_image(encoded, "image/gif")

        self.assertEqual(raised.exception.status_code, 415)

    def test_decoded_size_limit_is_enforced_before_analysis(self) -> None:
        oversized = b"\xff\xd8\xff" + b"x" * (1024 * 1024)
        encoded = base64.b64encode(oversized).decode("ascii")

        with (
            patch("api.image_uploads.MAX_FRAME_SIZE_MB", 1),
            self.assertRaises(HTTPException) as raised,
        ):
            decode_base64_image(encoded, "image/jpeg")

        self.assertEqual(raised.exception.status_code, 413)


if __name__ == "__main__":
    unittest.main()
