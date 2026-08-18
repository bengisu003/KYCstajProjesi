"""Tests for the unified single-image document classification workflow."""

import unittest
from unittest.mock import patch

import numpy as np

from services.document_classification import classify_document


def _prepared_frame(*, valid: bool = True) -> dict[str, object]:
    return {
        "valid": valid,
        "card_detected": True,
        "image_size": {"width": 100, "height": 60},
        "capture_quality": {
            "blur_score": 100.0,
            "dark_ratio": 0.0,
            "bright_ratio": 0.0,
        },
        "reason_codes": [] if valid else ["IMAGE_TOO_BLURRY"],
        "card_crop": np.zeros((4, 6, 3), dtype=np.uint8),
    }


def _front_analysis(prepared: dict[str, object]) -> dict[str, object]:
    return {
        **prepared,
        "document_score": 0.8,
        "features": {"flag_red_ratio": 0.1},
        "normalized_components": {
            "flag_vibrancy": 1.0,
            "flag_red_fidelity": 1.0,
            "security_palette": 1.0,
            "purple_band": 1.0,
        },
    }


class DocumentClassificationTests(unittest.TestCase):
    def test_invalid_prepared_frame_returns_before_side_analysis(self) -> None:
        prepared = _prepared_frame(valid=False)
        with (
            patch(
                "services.document_classification.prepare_document_frame",
                return_value=prepared,
            ),
            patch(
                "services.document_classification.analyze_prepared_document_frame"
            ) as analyze,
        ):
            result = classify_document(b"frame")

        self.assertEqual(result["decision"], "retake_required")
        self.assertIsNone(result["detected_side"])
        analyze.assert_not_called()

    def test_front_real_candidate_is_authorized_for_hologram(self) -> None:
        prepared = _prepared_frame()
        front_analysis = _front_analysis(prepared)
        with (
            patch(
                "services.document_classification.prepare_document_frame",
                return_value=prepared,
            ),
            patch(
                "services.document_classification.analyze_prepared_document_frame",
                return_value=front_analysis,
            ) as analyze,
            patch(
                "services.document_classification.calculate_front_side_score",
                return_value=(0.9, {}),
            ),
            patch(
                "services.document_classification.save_document_card_crop",
                return_value={
                    "sample_id": "sample",
                    "card_crop": "cropped_images/real/front/sample.jpg",
                },
            ) as save_crop,
            patch(
                "services.document_classification.register_real_candidate",
                return_value="comparison-token",
            ) as register,
        ):
            result = classify_document(b"frame", "front.jpg")

        self.assertEqual(result["decision"], "real_candidate")
        self.assertEqual(result["detected_side"], "front")
        self.assertTrue(result["hologram_ready"])
        analyze.assert_called_once_with(prepared, "front")
        save_crop.assert_called_once()
        register.assert_called_once()

    def test_back_real_candidate_does_not_authorize_hologram(self) -> None:
        prepared = _prepared_frame()
        front_analysis = _front_analysis(prepared)
        back_analysis = {
            **prepared,
            "document_score": 0.8,
            "features": {"mrz_contrast": 50.0},
            "normalized_components": {"mrz_contrast": 1.0},
        }

        def analyze_by_side(
            _prepared: dict[str, object], side: str
        ) -> dict[str, object]:
            return front_analysis if side == "front" else back_analysis

        with (
            patch(
                "services.document_classification.prepare_document_frame",
                return_value=prepared,
            ),
            patch(
                "services.document_classification.analyze_prepared_document_frame",
                side_effect=analyze_by_side,
            ) as analyze,
            patch(
                "services.document_classification.calculate_front_side_score",
                return_value=(0.2, {}),
            ),
            patch(
                "services.document_classification.save_document_card_crop",
                return_value={
                    "sample_id": "sample",
                    "card_crop": "cropped_images/real/back/sample.jpg",
                },
            ),
            patch(
                "services.document_classification.register_real_candidate"
            ) as register,
        ):
            result = classify_document(b"frame", "back.jpg")

        self.assertEqual(result["decision"], "real_candidate")
        self.assertEqual(result["detected_side"], "back")
        self.assertFalse(result["hologram_ready"])
        self.assertEqual(analyze.call_count, 2)
        register.assert_not_called()

    def test_ambiguous_side_does_not_save_or_authorize(self) -> None:
        prepared = _prepared_frame()
        front_analysis = _front_analysis(prepared)
        with (
            patch(
                "services.document_classification.prepare_document_frame",
                return_value=prepared,
            ),
            patch(
                "services.document_classification.analyze_prepared_document_frame",
                return_value=front_analysis,
            ),
            patch(
                "services.document_classification.calculate_front_side_score",
                return_value=(0.75, {}),
            ),
            patch(
                "services.document_classification.save_document_card_crop"
            ) as save_crop,
            patch(
                "services.document_classification.register_real_candidate"
            ) as register,
        ):
            result = classify_document(b"frame")

        self.assertEqual(result["decision"], "retake_required")
        self.assertIsNone(result["detected_side"])
        save_crop.assert_not_called()
        register.assert_not_called()

    def test_front_photocopy_is_saved_without_hologram_authorization(self) -> None:
        prepared = _prepared_frame()
        front_analysis = _front_analysis(prepared)
        with (
            patch(
                "services.document_classification.prepare_document_frame",
                return_value=prepared,
            ),
            patch(
                "services.document_classification.analyze_prepared_document_frame",
                return_value=front_analysis,
            ),
            patch(
                "services.document_classification.calculate_front_side_score",
                return_value=(0.9, {}),
            ),
            patch(
                "services.document_classification.calculate_front_classifier_score",
                return_value=0.6,
            ),
            patch(
                "services.document_classification.save_document_card_crop",
                return_value={
                    "sample_id": "sample",
                    "card_crop": "cropped_images/photocopy/front/sample.jpg",
                },
            ) as save_crop,
            patch(
                "services.document_classification.register_real_candidate"
            ) as register,
        ):
            result = classify_document(b"frame", "front-copy.jpg")

        self.assertEqual(result["decision"], "photocopy_suspected")
        self.assertFalse(result["hologram_ready"])
        save_crop.assert_called_once_with(
            prepared["card_crop"], "photocopy", "front"
        )
        register.assert_not_called()

    def test_inconclusive_document_score_requires_retake(self) -> None:
        prepared = _prepared_frame()
        front_analysis = _front_analysis(prepared)
        with (
            patch(
                "services.document_classification.prepare_document_frame",
                return_value=prepared,
            ),
            patch(
                "services.document_classification.analyze_prepared_document_frame",
                return_value=front_analysis,
            ),
            patch(
                "services.document_classification.calculate_front_side_score",
                return_value=(0.9, {}),
            ),
            patch(
                "services.document_classification.calculate_front_classifier_score",
                return_value=0.72,
            ),
            patch(
                "services.document_classification.save_document_card_crop"
            ) as save_crop,
            patch(
                "services.document_classification.register_real_candidate"
            ) as register,
        ):
            result = classify_document(b"frame")

        self.assertEqual(result["decision"], "retake_required")
        self.assertEqual(result["detected_side"], "front")
        self.assertIn(
            "INCONCLUSIVE_FRONT_DOCUMENT_CLASSIFICATION",
            result["reason_codes"],
        )
        save_crop.assert_not_called()
        register.assert_not_called()

    def test_conclusive_result_without_crop_requires_retake(self) -> None:
        prepared = _prepared_frame()
        front_analysis = {**_front_analysis(prepared), "card_crop": None}
        with (
            patch(
                "services.document_classification.prepare_document_frame",
                return_value=prepared,
            ),
            patch(
                "services.document_classification.analyze_prepared_document_frame",
                return_value=front_analysis,
            ),
            patch(
                "services.document_classification.calculate_front_side_score",
                return_value=(0.9, {}),
            ),
            patch(
                "services.document_classification.save_document_card_crop"
            ) as save_crop,
            patch(
                "services.document_classification.register_real_candidate"
            ) as register,
        ):
            result = classify_document(b"frame")

        self.assertEqual(result["decision"], "retake_required")
        self.assertIn("CARD_CROP_UNAVAILABLE", result["reason_codes"])
        save_crop.assert_not_called()
        register.assert_not_called()


if __name__ == "__main__":
    unittest.main()
