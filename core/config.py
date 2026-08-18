"""Configure storage, document classification, and hologram analysis."""
# bütün eşikler ve ayarlar.
import os
from pathlib import Path
import re

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _positive_int_from_env(name: str, default: int) -> int:
    """Read a positive integer setting and fail fast on invalid configuration."""
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a positive integer.") from error
    if value < 1:
        raise RuntimeError(f"{name} must be a positive integer.")
    return value


def _boolean_from_env(name: str, default: bool) -> bool:
    """Read a strict boolean setting and fail fast on invalid configuration."""
    raw_value = os.getenv(name, str(default)).strip().lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean value.")


def _choice_from_env(name: str, default: str, choices: set[str]) -> str:
    """Read one case-insensitive choice and return its canonical spelling."""
    raw_value = os.getenv(name, default).strip()
    canonical_choices = {choice.lower(): choice for choice in choices}
    selected = canonical_choices.get(raw_value.lower())
    if selected is None:
        allowed = ", ".join(sorted(choices))
        raise RuntimeError(f"{name} must be one of: {allowed}.")
    return selected


def _non_empty_string_from_env(name: str, default: str) -> str:
    """Read a non-empty string setting."""
    value = os.getenv(name, default).strip()
    if not value:
        raise RuntimeError(f"{name} cannot be empty.")
    return value


def _project_subdirectory_from_env(name: str, default: str) -> Path:
    """Resolve a configured relative directory safely below the project root."""
    configured_path = Path(_non_empty_string_from_env(name, default))
    if configured_path.is_absolute():
        raise RuntimeError(f"{name} must be a relative project path.")
    resolved_path = (PROJECT_ROOT / configured_path).resolve()
    if resolved_path == PROJECT_ROOT or not resolved_path.is_relative_to(PROJECT_ROOT):
        raise RuntimeError(f"{name} must point to a subdirectory of the project.")
    return resolved_path


def _cookie_name_from_env(name: str, default: str) -> str:
    """Read an HTTP-token-compatible cookie name."""
    value = _non_empty_string_from_env(name, default)
    if re.fullmatch(r"[A-Za-z0-9!#$%&'*+\-.^_`|~]+", value) is None:
        raise RuntimeError(f"{name} contains invalid cookie-name characters.")
    return value


MODEL_VERSION = "baseline-0.13.0"
LOG_LEVEL = _choice_from_env(
    "LOG_LEVEL",
    "INFO",
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"},
)

# Analysis crop output. Raw uploads are never stored.
CROP_OUTPUT_DIR = _project_subdirectory_from_env(
    "CROP_OUTPUT_DIR", "cropped_images"
)
CROP_RETENTION_DAYS = _positive_int_from_env("CROP_RETENTION_DAYS", 30)
CROP_CLEANUP_INTERVAL_SECONDS = _positive_int_from_env(
    "CROP_CLEANUP_INTERVAL_SECONDS", 60 * 60
)

# Successful front-document checks authorize hologram analysis for a short time. The
# normalized real-front crop is cached in memory until one check or expiration.
DOCUMENT_SESSION_TTL_SECONDS = _positive_int_from_env(
    "DOCUMENT_SESSION_TTL_SECONDS", 15 * 60
)
MAX_DOCUMENT_SESSIONS = _positive_int_from_env(
    "MAX_DOCUMENT_SESSIONS", 32
)
HOLOGRAM_SESSION_COOKIE = _cookie_name_from_env(
    "HOLOGRAM_SESSION_COOKIE", "hologram_session"
)
COOKIE_SECURE = _boolean_from_env("COOKIE_SECURE", False)
COOKIE_SAMESITE = _choice_from_env(
    "COOKIE_SAMESITE", "strict", {"strict", "lax", "none"}
)

# Request safety limits.
MAX_FRAME_SIZE_MB = _positive_int_from_env("MAX_FRAME_SIZE_MB", 10)
MAX_FRAME_PIXELS = _positive_int_from_env("MAX_FRAME_PIXELS", 16_000_000)

# Card geometry and contour-detector thresholds. These are prototype values and
# must be calibrated on a broader labeled image dataset.
CARD_WIDTH = 856
CARD_HEIGHT = 540
EXPECTED_CARD_RATIO = 85.60 / 53.98
CARD_RATIO_TOLERANCE = 0.45
MIN_CARD_AREA_RATIO = 0.08
CARD_DETECTION_MAX_DIMENSION = 1280
CARD_DETECTION_FALLBACK_MAX_DIMENSION = 1920
CARD_CROP_PADDING_PIXELS = 5.0
# Crop experiments can be reverted instantly by changing this to "baseline".
# Improved-mode values are initial calibration values for the current tests.
CARD_DETECTION_MODE = "improved"
CARD_CROP_PADDING_RATIO = 0.008
CARD_CROP_PADDING_MAX_PIXELS = 18.0
CARD_CORNER_REFINEMENT_WINDOW = 15
CARD_CORNER_MAX_SHIFT_RATIO = 0.025
MIN_IMPROVED_CARD_AREA_RATIO = 0.06
# Improved-mode recovery thresholds. A Lab color mask is consulted only when
# the original contour is geometrically suspicious. These prototype values
# must be recalibrated on a broader labeled crop dataset.
CARD_RECOVERY_TINY_AREA_RATIO = 0.08
CARD_RECOVERY_MIN_RATIO = 1.28
CARD_RECOVERY_MAX_RATIO = 1.88
CARD_RECOVERY_BORDER_MARGIN_RATIO = 0.012
CARD_RECOVERY_LARGE_AREA_RATIO = 0.55
CARD_RECOVERY_LARGE_AREA_MAX_RATIO = 1.45
CARD_RECOVERY_FRONT_RED_RATIO = 0.02
# Front-card flag-anchor recovery. This is consulted only after the normal
# contour and Lab recovery paths produce an unsafe oversized paper crop.
# These initial values must be recalibrated on broader front-card captures.
CARD_FLAG_ANCHOR_MIN_COMPONENT_AREA_RATIO = 0.0005
CARD_FLAG_ANCHOR_WIDTH_SCALE = 4.15
CARD_FLAG_ANCHOR_HEIGHT_SCALE = 3.25
CARD_FLAG_ANCHOR_CENTER_X_RATIO = 0.75
CARD_FLAG_ANCHOR_CENTER_Y_RATIO = 0.30
MAX_FALLBACK_CARD_AREA_RATIO = 0.45
# Internal contours are a last-resort detector. A stricter minimum prevents
# portrait/photo regions inside the identity card from being treated as a card.
MIN_INTERNAL_CARD_AREA_RATIO = 0.12
MAX_INTERNAL_CARD_AREA_RATIO = 0.65

# Single-frame capture-quality thresholds.
MIN_BLUR_SCORE = 45.0
MAX_DARK_PIXEL_RATIO = 0.40
MAX_BRIGHT_PIXEL_RATIO = 0.40

# Static hologram ROI and feature normalization ranges. These values were only
# pre-adjusted on the current labeled image pair and are not production values.
ROI_SIZE = 256
STATIC_ROI_X1 = 0.22
STATIC_ROI_Y1 = 0.38
STATIC_ROI_X2 = 0.50
STATIC_ROI_Y2 = 0.95
STATIC_DETAIL_LOW = 3000.0
STATIC_DETAIL_HIGH = 6500.0
STATIC_CONTRAST_LOW = 40.0
STATIC_CONTRAST_HIGH = 75.0
STATIC_COLORFUL_RATIO_LOW = 0.002
STATIC_COLORFUL_RATIO_HIGH = 0.065
STATIC_HIGHLIGHT_RATIO_LOW = 0.001
STATIC_HIGHLIGHT_RATIO_HIGH = 0.020
STATIC_LOCAL_HIGHLIGHT_LOW = 0.0005
STATIC_LOCAL_HIGHLIGHT_HIGH = 0.010
STATIC_HUE_ENTROPY_LOW = 0.10
STATIC_HUE_ENTROPY_HIGH = 0.75
STATIC_BRIGHT_AREA_LOW = 0.08
STATIC_BRIGHT_AREA_HIGH = 0.38
STATIC_PEAK_CHROMA_LOW = 16.0
STATIC_PEAK_CHROMA_HIGH = 32.0
STATIC_HOLOGRAM_ABSENT_MAX_SCORE = 0.15
STATIC_HOLOGRAM_PRESENT_MIN_SCORE = 0.30

# Single-image side-classification limits.
MIN_FRONT_SIDE_ANCHOR_SCORE = 0.80
MAX_BACK_SIDE_ANCHOR_SCORE = 0.70

# Front-side fixed color ROIs. Personal portrait/text fields are deliberately
# excluded as much as possible. The ranges below were calibrated on the current
# 50 real/photocopy front-image pairs and therefore require validation on new,
# unseen cards and lighting conditions before production use.
FRONT_FLAG_X1 = 0.58
FRONT_FLAG_Y1 = 0.14
FRONT_FLAG_X2 = 0.96
FRONT_FLAG_Y2 = 0.59
FRONT_PURPLE_BAND_X1 = 0.32
FRONT_PURPLE_BAND_Y1 = 0.70
FRONT_PURPLE_BAND_X2 = 0.95
FRONT_PURPLE_BAND_Y2 = 0.95
FRONT_SECURITY_X1 = 0.38
FRONT_SECURITY_Y1 = 0.08
FRONT_SECURITY_X2 = 0.78
FRONT_SECURITY_Y2 = 0.68

FRONT_FLAG_SATURATION_LOW = 16.0
FRONT_FLAG_SATURATION_HIGH = 23.0
FRONT_FLAG_CHROMA_LOW = 4.5
FRONT_FLAG_CHROMA_HIGH = 6.5
FRONT_FLAG_RED_RATIO_LOW = 0.02
FRONT_FLAG_RED_RATIO_HIGH = 0.09
FRONT_FLAG_RED_SATURATION_LOW = 60.0
FRONT_FLAG_RED_SATURATION_HIGH = 85.0
FRONT_SECURITY_SATURATION_LOW = 17.0
FRONT_SECURITY_SATURATION_HIGH = 22.0
FRONT_SECURITY_CHROMA_LOW = 4.3
FRONT_SECURITY_CHROMA_HIGH = 5.8
FRONT_SECURITY_HUE_ENTROPY_LOW = 0.60
FRONT_SECURITY_HUE_ENTROPY_HIGH = 0.75
FRONT_PURPLE_BAND_CHROMA_LOW = 3.4
FRONT_PURPLE_BAND_CHROMA_HIGH = 4.4
FRONT_PURPLE_BAND_COLORFUL_LOW = 0.006
FRONT_PURPLE_BAND_COLORFUL_HIGH = 0.020

# Front-side real/photocopy risk classifier. The feature normalizers above
# are shared by the single-image flows, while these weights and decision limits
# were calibrated on 35
# labeled real/photocopy pairs; 15 reserved pairs must remain evaluation-only.
FRONT_CLASSIFIER_FLAG_VIBRANCY_WEIGHT = 0.20
FRONT_CLASSIFIER_FLAG_RED_FIDELITY_WEIGHT = 0.40
FRONT_CLASSIFIER_SECURITY_PALETTE_WEIGHT = 0.35
FRONT_CLASSIFIER_PURPLE_BAND_WEIGHT = 0.05
FRONT_CLASSIFIER_PHOTOCOPY_MAX_SCORE = 0.68
FRONT_CLASSIFIER_REAL_MIN_SCORE = 0.76

# Back-side classification uses the MRZ and the full-card print/color response
# instead of the front-side portrait and hologram-bearing layout. These are
# baseline values derived from the first back-side test pair; they must be
# calibrated on the complete labeled real/photocopy test set.
BACK_MRZ_X1 = 0.04
BACK_MRZ_Y1 = 0.63
BACK_MRZ_X2 = 0.96
BACK_MRZ_Y2 = 0.97
BACK_MRZ_CONTRAST_LOW = 28.0
BACK_MRZ_CONTRAST_HIGH = 50.0
BACK_MRZ_DETAIL_LOW = 900.0
BACK_MRZ_DETAIL_HIGH = 1700.0
BACK_SATURATION_MEAN_LOW = 8.0
BACK_SATURATION_MEAN_HIGH = 18.0
BACK_EDGE_DENSITY_LOW = 0.12
BACK_EDGE_DENSITY_HIGH = 0.18
# Fixed back-side security-feature ROIs and prototype normalization ranges.
# They must be validated on independent cards, printers, cameras, and lighting.
BACK_CHIP_X1 = 0.07
BACK_CHIP_Y1 = 0.28
BACK_CHIP_X2 = 0.31
BACK_CHIP_Y2 = 0.62
BACK_CHIP_SATURATION_LOW = 22.0
BACK_CHIP_SATURATION_HIGH = 30.0
BACK_CHIP_CONTRAST_LOW = 40.0
BACK_CHIP_CONTRAST_HIGH = 52.0
BACK_CHIP_DETAIL_LOW = 1800.0
BACK_CHIP_DETAIL_HIGH = 4500.0
BACK_DOVID_X1 = 0.80
BACK_DOVID_Y1 = 0.10
BACK_DOVID_X2 = 0.96
BACK_DOVID_Y2 = 0.31
BACK_DOVID_RED_RATIO_LOW = 0.005
BACK_DOVID_RED_RATIO_HIGH = 0.10
BACK_DOVID_SATURATION_LOW = 18.0
BACK_DOVID_SATURATION_HIGH = 28.0
BACK_DOVID_CONTRAST_LOW = 40.0
BACK_DOVID_CONTRAST_HIGH = 52.0
BACK_BUILDING_X1 = 0.42
BACK_BUILDING_Y1 = 0.27
BACK_BUILDING_X2 = 0.83
BACK_BUILDING_Y2 = 0.60
BACK_BUILDING_SATURATION_LOW = 8.0
BACK_BUILDING_SATURATION_HIGH = 16.0
BACK_BUILDING_CHROMA_LOW = 2.5
BACK_BUILDING_CHROMA_HIGH = 5.5
BACK_BUILDING_DETAIL_LOW = 3000.0
BACK_BUILDING_DETAIL_HIGH = 6500.0
BACK_TRANSITION_DELTA_LOW = 5.0
BACK_TRANSITION_DELTA_HIGH = 10.0
BACK_TRANSITION_PEAK_TARGET = 4.5
BACK_TRANSITION_DEVIATION_LOW = 0.6
BACK_TRANSITION_DEVIATION_HIGH = 3.0
BACK_CLASSIFIER_PHOTOCOPY_MAX_SCORE = 0.65
BACK_CLASSIFIER_REAL_MIN_SCORE = 0.70
