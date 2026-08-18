"""Tests for the unified document response builder and public model."""

import unittest

from api.models import DocumentCheckResponse
from core.config import DOCUMENT_SESSION_TTL_SECONDS
from services.document.response_builder import (
    build_document_classification_response,
)


def _analysis() -> dict[str, object]:
    return {
        "card_detected": True,
        "image_size": {"width": 1280, "height": 720},
        "capture_quality": {
            "blur_score": 123.456789,
            "dark_ratio": 0.01,
            "bright_ratio": 0.02,
        },
        "features": {"signal": 0.123456789},
        "normalized_components": {"signal": 0.987654321},
    }


class DocumentResponseBuilderTests(unittest.TestCase):
    def test_front_real_response_matches_public_model(self) -> None:
        payload = build_document_classification_response(
            analysis=_analysis(),
            decision="real_candidate",
            detected_side="front",
            front_side_score=0.912345,
            document_score=0.87654321,
            classification_thresholds={
                "photocopy_max": 0.68,
                "real_min": 0.76,
            },
            reason_codes=["DOCUMENT_FRONT_SIDE_DETECTED"],
            saved_crop={
                "sample_id": "sample-1",
                "card_crop": "cropped_images/real/front/sample-1.jpg",
            },
            document_session_id="document-session-1",
            source_filename="card.jpg",
        )

        response = DocumentCheckResponse(**payload)

        self.assertEqual(response.detected_side, "front")
        self.assertEqual(response.front_side_score, 0.9123)
        self.assertEqual(response.document_score, 0.876543)
        self.assertEqual(response.features, {"signal": 0.1235})
        self.assertTrue(response.hologram_ready)
        self.assertEqual(
            response.hologram_session_expires_in_seconds,
            DOCUMENT_SESSION_TTL_SECONDS,
        )
        self.assertEqual(response.document_session_id, "document-session-1")

    def test_retake_response_has_no_hologram_metadata(self) -> None:
        payload = build_document_classification_response(
            analysis=_analysis(),
            decision="retake_required",
            detected_side=None,
            front_side_score=0.75,
            document_score=None,
            classification_thresholds=None,
            reason_codes=["DOCUMENT_SIDE_AMBIGUOUS"],
        )

        response = DocumentCheckResponse(**payload)

        self.assertFalse(response.hologram_ready)
        self.assertIsNone(response.document_session_id)
        self.assertIsNone(response.hologram_session_expires_in_seconds)
        self.assertIsNone(response.saved_crop)


if __name__ == "__main__":
    unittest.main()
