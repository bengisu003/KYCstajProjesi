"""Extract and score static hologram evidence features."""

import cv2
import numpy as np

from core.config import (
    ROI_SIZE,
    STATIC_BRIGHT_AREA_HIGH,
    STATIC_BRIGHT_AREA_LOW,
    STATIC_COLORFUL_RATIO_HIGH,
    STATIC_COLORFUL_RATIO_LOW,
    STATIC_CONTRAST_HIGH,
    STATIC_CONTRAST_LOW,
    STATIC_DETAIL_HIGH,
    STATIC_DETAIL_LOW,
    STATIC_HIGHLIGHT_RATIO_HIGH,
    STATIC_HIGHLIGHT_RATIO_LOW,
    STATIC_HUE_ENTROPY_HIGH,
    STATIC_HUE_ENTROPY_LOW,
    STATIC_LOCAL_HIGHLIGHT_HIGH,
    STATIC_LOCAL_HIGHLIGHT_LOW,
    STATIC_PEAK_CHROMA_HIGH,
    STATIC_PEAK_CHROMA_LOW,
    STATIC_ROI_X1,
    STATIC_ROI_X2,
    STATIC_ROI_Y1,
    STATIC_ROI_Y2,
)
from infrastructure.vision.card import normalize_feature


def extract_static_hologram_roi(warped_card: np.ndarray) -> np.ndarray:
    """Crop the tighter physical hologram area used for single-frame evidence."""
    height, width = warped_card.shape[:2]
    x1 = int(np.clip(round(STATIC_ROI_X1 * width), 0, width))
    y1 = int(np.clip(round(STATIC_ROI_Y1 * height), 0, height))
    x2 = int(np.clip(round(STATIC_ROI_X2 * width), 0, width))
    y2 = int(np.clip(round(STATIC_ROI_Y2 * height), 0, height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Static hologram ROI koordinatlari gecersiz.")
    roi = warped_card[y1:y2, x1:x2]
    if roi.size == 0:
        raise ValueError("Static hologram ROI alani bos.")
    return cv2.resize(roi, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_AREA)


def extract_static_hologram_features(roi: np.ndarray) -> dict[str, float]:
    """Extract localized reflection and pattern-detail evidence from one ROI."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    chroma_distance = np.linalg.norm(lab[:, :, 1:3] - 128.0, axis=2)
    colorful_mask = (saturation > 60) & (value > 100)
    highlight_mask = (saturation > 70) & (value > 180)

    colorful_ratio = float(np.mean(colorful_mask))
    highlight_ratio = float(np.mean(highlight_mask))
    connected_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        highlight_mask.astype(np.uint8),
        connectivity=8,
    )
    largest_highlight_area = (
        float(np.max(stats[1:, cv2.CC_STAT_AREA])) if connected_count > 1 else 0.0
    )
    largest_highlight_ratio = largest_highlight_area / float(highlight_mask.size)

    hue_entropy = 0.0
    if np.any(colorful_mask):
        histogram = cv2.calcHist(
            [hsv], [0], colorful_mask.astype(np.uint8), [18], [0, 180]
        ).reshape(-1)
        histogram /= max(float(np.sum(histogram)), 1.0)
        nonzero = histogram[histogram > 0]
        hue_entropy = float(-np.sum(nonzero * np.log(nonzero)) / np.log(18.0))

    return {
        "detail_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "local_contrast": float(np.std(gray)),
        "colorful_ratio": colorful_ratio,
        "highlight_ratio": highlight_ratio,
        "largest_highlight_ratio": largest_highlight_ratio,
        "hue_entropy": hue_entropy,
        "bright_area_ratio": float(np.mean(value > 190)),
        "peak_chroma": float(np.percentile(chroma_distance, 99)),
    }


def calculate_static_hologram_score(
    features: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """Combine static single-frame hologram evidence into a bounded score."""
    components = {
        "pattern_detail": normalize_feature(
            features["detail_variance"], STATIC_DETAIL_LOW, STATIC_DETAIL_HIGH
        ),
        "local_contrast": normalize_feature(
            features["local_contrast"], STATIC_CONTRAST_LOW, STATIC_CONTRAST_HIGH
        ),
        "colorful_area": normalize_feature(
            features["colorful_ratio"],
            STATIC_COLORFUL_RATIO_LOW,
            STATIC_COLORFUL_RATIO_HIGH,
        ),
        "colored_highlight": normalize_feature(
            features["highlight_ratio"],
            STATIC_HIGHLIGHT_RATIO_LOW,
            STATIC_HIGHLIGHT_RATIO_HIGH,
        ),
        "localized_highlight": normalize_feature(
            features["largest_highlight_ratio"],
            STATIC_LOCAL_HIGHLIGHT_LOW,
            STATIC_LOCAL_HIGHLIGHT_HIGH,
        ),
        "hue_diversity": normalize_feature(
            features["hue_entropy"], STATIC_HUE_ENTROPY_LOW, STATIC_HUE_ENTROPY_HIGH
        ),
        "bright_area": normalize_feature(
            features["bright_area_ratio"],
            STATIC_BRIGHT_AREA_LOW,
            STATIC_BRIGHT_AREA_HIGH,
        ),
        "peak_chroma": normalize_feature(
            features["peak_chroma"],
            STATIC_PEAK_CHROMA_LOW,
            STATIC_PEAK_CHROMA_HIGH,
        ),
    }
    score = (
        0.05 * components["pattern_detail"]
        + 0.10 * components["local_contrast"]
        + 0.15 * components["colorful_area"]
        + 0.10 * components["colored_highlight"]
        + 0.10 * components["localized_highlight"]
        + 0.10 * components["hue_diversity"]
        + 0.25 * components["bright_area"]
        + 0.15 * components["peak_chroma"]
    )
    return float(np.clip(score, 0.0, 1.0)), components
