"""Tests for the shared front-side scoring policy."""

import unittest

from services.document.front_classifier import (
    COMPONENT_WEIGHTS,
    calculate_front_classifier_score,
)


class FrontClassifierTests(unittest.TestCase):
    def test_component_weights_form_a_complete_score(self) -> None:
        self.assertAlmostEqual(sum(COMPONENT_WEIGHTS.values()), 1.0)

    def test_score_respects_zero_and_one_boundaries(self) -> None:
        zero_components = {name: 0.0 for name in COMPONENT_WEIGHTS}
        one_components = {name: 1.0 for name in COMPONENT_WEIGHTS}
        self.assertEqual(calculate_front_classifier_score(zero_components), 0.0)
        self.assertEqual(calculate_front_classifier_score(one_components), 1.0)

if __name__ == "__main__":
    unittest.main()
