"""Tests for short-lived, single-use hologram authorizations."""

import unittest

import numpy as np

from core.exceptions import DocumentSessionAuthorizationError
from infrastructure.session.hologram_authorization import (
    consume_real_candidate,
    register_real_candidate,
)


class HologramSessionTests(unittest.TestCase):
    def test_missing_session_is_rejected_with_reason_code(self) -> None:
        with self.assertRaises(DocumentSessionAuthorizationError) as raised:
            consume_real_candidate("")
        self.assertEqual(
            raised.exception.reason_code,
            "DOCUMENT_SESSION_REQUIRED",
        )

    def test_cached_candidate_is_copied_and_consumed_once(self) -> None:
        original_crop = np.zeros((2, 3, 3), dtype=np.uint8)
        document_session_id = register_real_candidate(
            "synthetic-sample",
            original_crop,
            {"width": 3, "height": 2},
            "frame_1",
            "synthetic.jpg",
        )
        original_crop[:] = 255

        crop, image_size, sample_id, source_frame, source_filename = (
            consume_real_candidate(document_session_id)
        )
        self.assertEqual(int(crop.max()), 0)
        self.assertEqual(image_size, {"width": 3, "height": 2})
        self.assertEqual(sample_id, "synthetic-sample")
        self.assertEqual(source_frame, "frame_1")
        self.assertEqual(source_filename, "synthetic.jpg")

        with self.assertRaises(DocumentSessionAuthorizationError) as raised:
            consume_real_candidate(document_session_id)
        self.assertEqual(
            raised.exception.reason_code,
            "DOCUMENT_SESSION_REQUIRED",
        )


if __name__ == "__main__":
    unittest.main()
