"""Simple image quality heuristics for pet face crops."""

from __future__ import annotations

import cv2
import numpy as np


def is_sharp_enough(bgr: np.ndarray, thresh: float = 100.0) -> bool:
    """Return True if the Laplacian variance meets the sharpness threshold."""
    if bgr is None or bgr.ndim != 3:
        raise ValueError("Input must be an HxWx3 BGR image.")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var >= thresh


def is_big_enough(bgr: np.ndarray, min_side: int = 120) -> bool:
    """Return True if the smallest image side meets the minimum requirement."""
    if bgr is None or bgr.ndim != 3:
        raise ValueError("Input must be an HxWx3 BGR image.")
    h, w = bgr.shape[:2]
    return min(h, w) >= min_side


def pass_quality(bgr: np.ndarray, lap_thresh: float = 100.0, min_side: int = 120) -> bool:
    """Return True if the image passes both sharpness and size checks."""
    return is_sharp_enough(bgr, lap_thresh) and is_big_enough(bgr, min_side)
