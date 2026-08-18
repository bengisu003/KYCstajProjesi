"""Tests for absolute front/back real-photocopy decision policies."""

import unittest

from services.document.classification_policy import (
    classify_document_score,
    get_document_classification_thresholds,
)


class DocumentClassificationPolicyTests(unittest.TestCase):
    def test_front_score_boundaries(self) -> None:
        self.assertEqual(
            classify_document_score("front", 0.68)[0], "photocopy_suspected"
        )
        self.assertEqual(
            classify_document_score("front", 0.76)[0], "real_candidate"
        )
        self.assertEqual(
            classify_document_score("front", 0.72)[0], "retake_required"
        )

    def test_back_score_boundaries(self) -> None:
        self.assertEqual(
            classify_document_score("back", 0.65)[0], "photocopy_suspected"
        )
        self.assertEqual(
            classify_document_score("back", 0.70)[0], "real_candidate"
        )
        self.assertEqual(
            classify_document_score("back", 0.68)[0], "retake_required"
        )

    def test_thresholds_are_returned_per_side(self) -> None:
        self.assertEqual(
            get_document_classification_thresholds("front"),
            {"photocopy_max_score": 0.68, "real_min_score": 0.76},
        )
        self.assertEqual(
            get_document_classification_thresholds("back"),
            {"photocopy_max_score": 0.65, "real_min_score": 0.70},
        )

    def test_invalid_score_is_rejected(self) -> None:
        for score in (-0.01, 1.01, float("nan")):
            with self.subTest(score=score):
                with self.assertRaises(ValueError):
                    classify_document_score("front", score)


if __name__ == "__main__":
    unittest.main()
