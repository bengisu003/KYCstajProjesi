"""Calculate calibrated front-side real/photocopy evidence scores."""

import math

from core.config import (
    FRONT_CLASSIFIER_FLAG_RED_FIDELITY_WEIGHT,
    FRONT_CLASSIFIER_FLAG_VIBRANCY_WEIGHT,
    FRONT_CLASSIFIER_PHOTOCOPY_MAX_SCORE,
    FRONT_CLASSIFIER_PURPLE_BAND_WEIGHT,
    FRONT_CLASSIFIER_REAL_MIN_SCORE,
    FRONT_CLASSIFIER_SECURITY_PALETTE_WEIGHT,
)


COMPONENT_WEIGHTS = {
    "flag_vibrancy": FRONT_CLASSIFIER_FLAG_VIBRANCY_WEIGHT,
    "flag_red_fidelity": FRONT_CLASSIFIER_FLAG_RED_FIDELITY_WEIGHT,
    "security_palette": FRONT_CLASSIFIER_SECURITY_PALETTE_WEIGHT,
    "purple_band": FRONT_CLASSIFIER_PURPLE_BAND_WEIGHT,
}

if not math.isclose(sum(COMPONENT_WEIGHTS.values()), 1.0, abs_tol=1e-9):
    raise RuntimeError("Front-side classifier component weights must total 1.0.")
if FRONT_CLASSIFIER_PHOTOCOPY_MAX_SCORE >= FRONT_CLASSIFIER_REAL_MIN_SCORE:
    raise RuntimeError("Front-side classifier decision thresholds are invalid.")


def calculate_front_classifier_score(components: dict[str, float]) -> float:
    """Combine normalized front-card components with calibrated weights."""
    score = sum(
        COMPONENT_WEIGHTS[name] * float(components[name])
        for name in COMPONENT_WEIGHTS
    )
    return float(min(max(score, 0.0), 1.0))
