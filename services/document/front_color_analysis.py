"""Extract and score front-side color evidence from normalized cards."""

import cv2
import numpy as np

from core.config import (
    FRONT_FLAG_CHROMA_HIGH,
    FRONT_FLAG_CHROMA_LOW,
    FRONT_FLAG_RED_RATIO_HIGH,
    FRONT_FLAG_RED_RATIO_LOW,
    FRONT_FLAG_RED_SATURATION_HIGH,
    FRONT_FLAG_RED_SATURATION_LOW,
    FRONT_FLAG_SATURATION_HIGH,
    FRONT_FLAG_SATURATION_LOW,
    FRONT_FLAG_X1,
    FRONT_FLAG_X2,
    FRONT_FLAG_Y1,
    FRONT_FLAG_Y2,
    FRONT_PURPLE_BAND_CHROMA_HIGH,
    FRONT_PURPLE_BAND_CHROMA_LOW,
    FRONT_PURPLE_BAND_COLORFUL_HIGH,
    FRONT_PURPLE_BAND_COLORFUL_LOW,
    FRONT_PURPLE_BAND_X1,
    FRONT_PURPLE_BAND_X2,
    FRONT_PURPLE_BAND_Y1,
    FRONT_PURPLE_BAND_Y2,
    FRONT_SECURITY_CHROMA_HIGH,
    FRONT_SECURITY_CHROMA_LOW,
    FRONT_SECURITY_HUE_ENTROPY_HIGH,
    FRONT_SECURITY_HUE_ENTROPY_LOW,
    FRONT_SECURITY_SATURATION_HIGH,
    FRONT_SECURITY_SATURATION_LOW,
    FRONT_SECURITY_X1,
    FRONT_SECURITY_X2,
    FRONT_SECURITY_Y1,
    FRONT_SECURITY_Y2,
)
from infrastructure.vision.card import normalize_feature
from services.document.analysis_helpers import extract_relative_roi


def _gray_world_balance(image: np.ndarray) -> np.ndarray:
    """Reduce camera white-balance differences without using personal data."""
    float_image = image.astype(np.float32)
    channel_means = np.mean(float_image, axis=(0, 1))
    target_mean = float(np.mean(channel_means))
    gains = target_mean / np.maximum(channel_means, 1.0)
    return np.clip(float_image * gains, 0.0, 255.0).astype(np.uint8)


def _normalized_hue_entropy(hue: np.ndarray, saturation: np.ndarray) -> float:
    """Measure hue spread while ignoring nearly neutral pixels."""
    selected_hues = hue[saturation > 12]
    if selected_hues.size == 0:
        return 0.0
    histogram, _ = np.histogram(selected_hues, bins=24, range=(0, 180))
    probabilities = histogram[histogram > 0].astype(np.float64)
    probabilities /= np.sum(probabilities)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy / np.log2(24.0))


def _front_color_region_metrics(region: np.ndarray) -> dict[str, float]:
    """Extract illumination-resistant HSV/Lab statistics from one fixed ROI."""
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).astype(np.float32)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    a_channel = lab[:, :, 1] - 128.0
    b_channel = lab[:, :, 2] - 128.0
    chroma = np.sqrt(a_channel * a_channel + b_channel * b_channel)
    return {
        "saturation_mean": float(np.mean(saturation)),
        "chroma_mean": float(np.mean(chroma)),
        "colorful_ratio": float(np.mean((saturation > 35) & (value > 80))),
        "hue_entropy": _normalized_hue_entropy(hsv[:, :, 0], saturation),
    }


def extract_front_color_features(
    warped_card: np.ndarray,
) -> dict[str, float]:
    """Measure fixed front-side flag, security-palette, and lower-band colors."""
    balanced_card = _gray_world_balance(warped_card)
    flag = extract_relative_roi(
        balanced_card,
        FRONT_FLAG_X1,
        FRONT_FLAG_Y1,
        FRONT_FLAG_X2,
        FRONT_FLAG_Y2,
    )
    purple_band = extract_relative_roi(
        balanced_card,
        FRONT_PURPLE_BAND_X1,
        FRONT_PURPLE_BAND_Y1,
        FRONT_PURPLE_BAND_X2,
        FRONT_PURPLE_BAND_Y2,
    )
    security = extract_relative_roi(
        balanced_card,
        FRONT_SECURITY_X1,
        FRONT_SECURITY_Y1,
        FRONT_SECURITY_X2,
        FRONT_SECURITY_Y2,
    )

    flag_metrics = _front_color_region_metrics(flag)
    band_metrics = _front_color_region_metrics(purple_band)
    security_metrics = _front_color_region_metrics(security)
    flag_hsv = cv2.cvtColor(flag, cv2.COLOR_BGR2HSV)
    red_mask = (
        ((flag_hsv[:, :, 0] < 12) | (flag_hsv[:, :, 0] > 170))
        & (flag_hsv[:, :, 1] > 35)
        & (flag_hsv[:, :, 2] > 70)
    )
    red_saturation = (
        float(np.mean(flag_hsv[:, :, 1][red_mask]))
        if np.any(red_mask)
        else 0.0
    )
    return {
        "flag_saturation_mean": flag_metrics["saturation_mean"],
        "flag_chroma_mean": flag_metrics["chroma_mean"],
        "flag_red_ratio": float(np.mean(red_mask)),
        "flag_red_saturation_mean": red_saturation,
        "security_saturation_mean": security_metrics["saturation_mean"],
        "security_chroma_mean": security_metrics["chroma_mean"],
        "security_hue_entropy": security_metrics["hue_entropy"],
        "purple_band_chroma_mean": band_metrics["chroma_mean"],
        "purple_band_colorful_ratio": band_metrics["colorful_ratio"],
    }


def calculate_front_color_score(
    features: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """Combine explainable front-side color-scale evidence."""
    flag_vibrancy = 0.50 * normalize_feature(
        features["flag_saturation_mean"],
        FRONT_FLAG_SATURATION_LOW,
        FRONT_FLAG_SATURATION_HIGH,
    ) + 0.50 * normalize_feature(
        features["flag_chroma_mean"],
        FRONT_FLAG_CHROMA_LOW,
        FRONT_FLAG_CHROMA_HIGH,
    )
    flag_red_fidelity = 0.70 * normalize_feature(
        features["flag_red_ratio"],
        FRONT_FLAG_RED_RATIO_LOW,
        FRONT_FLAG_RED_RATIO_HIGH,
    ) + 0.30 * normalize_feature(
        features["flag_red_saturation_mean"],
        FRONT_FLAG_RED_SATURATION_LOW,
        FRONT_FLAG_RED_SATURATION_HIGH,
    )
    security_palette = (
        0.30
        * normalize_feature(
            features["security_saturation_mean"],
            FRONT_SECURITY_SATURATION_LOW,
            FRONT_SECURITY_SATURATION_HIGH,
        )
        + 0.25
        * normalize_feature(
            features["security_chroma_mean"],
            FRONT_SECURITY_CHROMA_LOW,
            FRONT_SECURITY_CHROMA_HIGH,
        )
        + 0.45
        * (
            1.0
            - normalize_feature(
                features["security_hue_entropy"],
                FRONT_SECURITY_HUE_ENTROPY_LOW,
                FRONT_SECURITY_HUE_ENTROPY_HIGH,
            )
        )
    )
    purple_band = 0.60 * normalize_feature(
        features["purple_band_chroma_mean"],
        FRONT_PURPLE_BAND_CHROMA_LOW,
        FRONT_PURPLE_BAND_CHROMA_HIGH,
    ) + 0.40 * normalize_feature(
        features["purple_band_colorful_ratio"],
        FRONT_PURPLE_BAND_COLORFUL_LOW,
        FRONT_PURPLE_BAND_COLORFUL_HIGH,
    )
    components = {
        "flag_vibrancy": float(flag_vibrancy),
        "flag_red_fidelity": float(flag_red_fidelity),
        "security_palette": float(security_palette),
        "purple_band": float(purple_band),
    }
    score = (
        0.35 * components["flag_vibrancy"]
        + 0.25 * components["flag_red_fidelity"]
        + 0.30 * components["security_palette"]
        + 0.10 * components["purple_band"]
    )
    return float(np.clip(score, 0.0, 1.0)), components
