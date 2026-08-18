"""Tests for absolute front/back classification of one card image."""

import unittest

from services.document.side_detection import (
    classify_single_document_side,
    get_single_document_side_thresholds,
)


class SingleDocumentSideTests(unittest.TestCase):
    def test_front_and_back_boundaries_are_inclusive(self) -> None:
        self.assertEqual(classify_single_document_side(0.80)[0], "front")
        self.assertEqual(classify_single_document_side(0.70)[0], "back")

    def test_gap_requires_a_retake(self) -> None:
        side, reason_codes = classify_single_document_side(0.75)
        self.assertIsNone(side)
        self.assertEqual(reason_codes, ["AMBIGUOUS_DOCUMENT_SIDE"])

    def test_unavailable_score_is_not_treated_as_back(self) -> None:
        for score in (-1.0, 1.01, float("nan")):
            with self.subTest(score=score):
                side, reason_codes = classify_single_document_side(score)
                self.assertIsNone(side)
                self.assertEqual(
                    reason_codes,
                    ["DOCUMENT_SIDE_FEATURES_UNAVAILABLE"],
                )

    def test_threshold_metadata_matches_policy(self) -> None:
        self.assertEqual(
            get_single_document_side_thresholds(),
            {"front_min_score": 0.80, "back_max_score": 0.70},
        )


if __name__ == "__main__":
    unittest.main()
