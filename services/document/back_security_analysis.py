"""Extract and score back-side security evidence from normalized cards."""

import cv2
import numpy as np

from core.config import (
    BACK_BUILDING_CHROMA_HIGH,
    BACK_BUILDING_CHROMA_LOW,
    BACK_BUILDING_DETAIL_HIGH,
    BACK_BUILDING_DETAIL_LOW,
    BACK_BUILDING_SATURATION_HIGH,
    BACK_BUILDING_SATURATION_LOW,
    BACK_BUILDING_X1,
    BACK_BUILDING_X2,
    BACK_BUILDING_Y1,
    BACK_BUILDING_Y2,
    BACK_CHIP_CONTRAST_HIGH,
    BACK_CHIP_CONTRAST_LOW,
    BACK_CHIP_DETAIL_HIGH,
    BACK_CHIP_DETAIL_LOW,
    BACK_CHIP_SATURATION_HIGH,
    BACK_CHIP_SATURATION_LOW,
    BACK_CHIP_X1,
    BACK_CHIP_X2,
    BACK_CHIP_Y1,
    BACK_CHIP_Y2,
    BACK_DOVID_CONTRAST_HIGH,
    BACK_DOVID_CONTRAST_LOW,
    BACK_DOVID_RED_RATIO_HIGH,
    BACK_DOVID_RED_RATIO_LOW,
    BACK_DOVID_SATURATION_HIGH,
    BACK_DOVID_SATURATION_LOW,
    BACK_DOVID_X1,
    BACK_DOVID_X2,
    BACK_DOVID_Y1,
    BACK_DOVID_Y2,
    BACK_EDGE_DENSITY_HIGH,
    BACK_EDGE_DENSITY_LOW,
    BACK_MRZ_CONTRAST_HIGH,
    BACK_MRZ_CONTRAST_LOW,
    BACK_MRZ_DETAIL_HIGH,
    BACK_MRZ_DETAIL_LOW,
    BACK_MRZ_X1,
    BACK_MRZ_X2,
    BACK_MRZ_Y1,
    BACK_MRZ_Y2,
    BACK_SATURATION_MEAN_HIGH,
    BACK_SATURATION_MEAN_LOW,
    BACK_TRANSITION_DELTA_HIGH,
    BACK_TRANSITION_DELTA_LOW,
    BACK_TRANSITION_DEVIATION_HIGH,
    BACK_TRANSITION_DEVIATION_LOW,
    BACK_TRANSITION_PEAK_TARGET,
)
from infrastructure.vision.card import normalize_feature
from services.document.analysis_helpers import extract_relative_roi


def extract_back_document_features(
    warped_card: np.ndarray,
) -> dict[str, float]:
    """Measure back-side MRZ, color, and print-detail evidence."""
    gray = cv2.cvtColor(warped_card, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(warped_card, cv2.COLOR_BGR2HSV)
    height, width = gray.shape
    x1 = int(round(BACK_MRZ_X1 * width))
    y1 = int(round(BACK_MRZ_Y1 * height))
    x2 = int(round(BACK_MRZ_X2 * width))
    y2 = int(round(BACK_MRZ_Y2 * height))
    mrz = gray[y1:y2, x1:x2]
    if mrz.size == 0:
        raise ValueError("Configured back-side MRZ region is empty.")

    chip = extract_relative_roi(
        warped_card,
        BACK_CHIP_X1,
        BACK_CHIP_Y1,
        BACK_CHIP_X2,
        BACK_CHIP_Y2,
    )
    dovid = extract_relative_roi(
        warped_card,
        BACK_DOVID_X1,
        BACK_DOVID_Y1,
        BACK_DOVID_X2,
        BACK_DOVID_Y2,
    )
    building = extract_relative_roi(
        warped_card,
        BACK_BUILDING_X1,
        BACK_BUILDING_Y1,
        BACK_BUILDING_X2,
        BACK_BUILDING_Y2,
    )

    def region_metrics(region: np.ndarray) -> dict[str, float]:
        region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        region_lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).astype(np.float32)
        chroma = np.linalg.norm(region_lab[:, :, 1:3] - 128.0, axis=2)
        red_mask = (
            ((region_hsv[:, :, 0] < 12) | (region_hsv[:, :, 0] > 170))
            & (region_hsv[:, :, 1] > 35)
            & (region_hsv[:, :, 2] > 60)
        )
        return {
            "saturation_mean": float(np.mean(region_hsv[:, :, 1])),
            "chroma_mean": float(np.mean(chroma)),
            "red_ratio": float(np.mean(red_mask)),
            "contrast": float(np.std(region_gray)),
            "detail_variance": float(
                cv2.Laplacian(region_gray, cv2.CV_64F).var()
            ),
        }

    chip_metrics = region_metrics(chip)
    dovid_metrics = region_metrics(dovid)
    building_metrics = region_metrics(building)
    card_lab = cv2.cvtColor(warped_card, cv2.COLOR_BGR2LAB).astype(np.float32)
    transition_top = extract_relative_roi(card_lab, 0.18, 0.10, 0.85, 0.38)
    transition_bottom = extract_relative_roi(card_lab, 0.18, 0.50, 0.85, 0.68)
    top_bottom_color_delta = float(
        np.linalg.norm(
            np.mean(transition_top, axis=(0, 1))
            - np.mean(transition_bottom, axis=(0, 1))
        )
    )
    transition_region = extract_relative_roi(card_lab, 0.18, 0.28, 0.88, 0.72)
    row_colors = np.mean(transition_region, axis=1)
    smoothed_rows = cv2.GaussianBlur(
        row_colors.reshape(-1, 1, 3),
        (1, 9),
        0,
    ).reshape(-1, 3)
    row_color_changes = np.linalg.norm(np.diff(smoothed_rows, axis=0), axis=1)
    transition_peak = float(np.percentile(row_color_changes, 95))

    edges = cv2.Canny(gray, 60, 160)
    return {
        "detail_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "local_contrast": float(np.std(gray)),
        "colorful_ratio": float(
            np.mean((hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 100))
        ),
        "edge_density": float(np.mean(edges > 0)),
        "mrz_contrast": float(np.std(mrz)),
        "mrz_detail_variance": float(
            cv2.Laplacian(mrz, cv2.CV_64F).var()
        ),
        "mean_saturation": float(np.mean(hsv[:, :, 1])),
        "chip_saturation_mean": chip_metrics["saturation_mean"],
        "chip_contrast": chip_metrics["contrast"],
        "chip_detail_variance": chip_metrics["detail_variance"],
        "dovid_red_ratio": dovid_metrics["red_ratio"],
        "dovid_saturation_mean": dovid_metrics["saturation_mean"],
        "dovid_contrast": dovid_metrics["contrast"],
        "building_saturation_mean": building_metrics["saturation_mean"],
        "building_chroma_mean": building_metrics["chroma_mean"],
        "building_detail_variance": building_metrics["detail_variance"],
        "top_bottom_color_delta": top_bottom_color_delta,
        "transition_peak": transition_peak,
    }


def calculate_back_document_score(
    features: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """Combine signals that are specific to the identity-card back side."""
    edge_noise = normalize_feature(
        features["edge_density"],
        BACK_EDGE_DENSITY_LOW,
        BACK_EDGE_DENSITY_HIGH,
    )
    chip_color = (
        0.40
        * normalize_feature(
            features["chip_saturation_mean"],
            BACK_CHIP_SATURATION_LOW,
            BACK_CHIP_SATURATION_HIGH,
        )
        + 0.35
        * normalize_feature(
            features["chip_contrast"],
            BACK_CHIP_CONTRAST_LOW,
            BACK_CHIP_CONTRAST_HIGH,
        )
        + 0.25
        * (
            1.0
            - normalize_feature(
                features["chip_detail_variance"],
                BACK_CHIP_DETAIL_LOW,
                BACK_CHIP_DETAIL_HIGH,
            )
        )
    )
    dovid_color = (
        0.55
        * normalize_feature(
            features["dovid_red_ratio"],
            BACK_DOVID_RED_RATIO_LOW,
            BACK_DOVID_RED_RATIO_HIGH,
        )
        + 0.25
        * normalize_feature(
            features["dovid_saturation_mean"],
            BACK_DOVID_SATURATION_LOW,
            BACK_DOVID_SATURATION_HIGH,
        )
        + 0.20
        * normalize_feature(
            features["dovid_contrast"],
            BACK_DOVID_CONTRAST_LOW,
            BACK_DOVID_CONTRAST_HIGH,
        )
    )
    building_visibility = (
        0.35
        * normalize_feature(
            features["building_saturation_mean"],
            BACK_BUILDING_SATURATION_LOW,
            BACK_BUILDING_SATURATION_HIGH,
        )
        + 0.45
        * normalize_feature(
            features["building_chroma_mean"],
            BACK_BUILDING_CHROMA_LOW,
            BACK_BUILDING_CHROMA_HIGH,
        )
        + 0.20
        * (
            1.0
            - normalize_feature(
                features["building_detail_variance"],
                BACK_BUILDING_DETAIL_LOW,
                BACK_BUILDING_DETAIL_HIGH,
            )
        )
    )
    transition_consistency = 1.0 - normalize_feature(
        abs(features["transition_peak"] - BACK_TRANSITION_PEAK_TARGET),
        BACK_TRANSITION_DEVIATION_LOW,
        BACK_TRANSITION_DEVIATION_HIGH,
    )
    color_transition = (
        0.70
        * normalize_feature(
            features["top_bottom_color_delta"],
            BACK_TRANSITION_DELTA_LOW,
            BACK_TRANSITION_DELTA_HIGH,
        )
        + 0.30 * transition_consistency
    )
    components = {
        "mrz_contrast": normalize_feature(
            features["mrz_contrast"],
            BACK_MRZ_CONTRAST_LOW,
            BACK_MRZ_CONTRAST_HIGH,
        ),
        "mrz_detail": normalize_feature(
            features["mrz_detail_variance"],
            BACK_MRZ_DETAIL_LOW,
            BACK_MRZ_DETAIL_HIGH,
        ),
        "color_integrity": normalize_feature(
            features["mean_saturation"],
            BACK_SATURATION_MEAN_LOW,
            BACK_SATURATION_MEAN_HIGH,
        ),
        "print_cleanliness": 1.0 - edge_noise,
        "chip_color": float(chip_color),
        "dovid_color": float(dovid_color),
        "building_visibility": float(building_visibility),
        "color_transition": float(color_transition),
    }
    score = (
        0.195 * components["mrz_contrast"]
        + 0.1625 * components["mrz_detail"]
        + 0.1625 * components["color_integrity"]
        + 0.13 * components["print_cleanliness"]
        + 0.10 * components["chip_color"]
        + 0.10 * components["dovid_color"]
        + 0.10 * components["building_visibility"]
        + 0.05 * components["color_transition"]
    )
    return float(np.clip(score, 0.0, 1.0)), components
