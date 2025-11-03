"""Utilities to align cat face crops using landmark files."""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np


CANVAS_SIZE = 224
LEFT_EYE_TARGET = np.array([80.0, 90.0])
RIGHT_EYE_TARGET = np.array([144.0, 90.0])
NUM_LANDMARKS = 9


def _ensure_points_array(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim == 1:
        if pts.size != NUM_LANDMARKS * 2:
            raise ValueError(f"Expected {NUM_LANDMARKS*2} values, got {pts.size}")
        pts = pts.reshape(NUM_LANDMARKS, 2)
    if pts.shape != (NUM_LANDMARKS, 2):
        raise ValueError(f"Landmarks must have shape (9, 2); got {pts.shape}")
    return pts


def align_face_bye2e(img_bgr: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Align a cat face crop to a canonical view using eye-to-eye similarity transform.

    Args:
        img_bgr: Source image in BGR format.
        pts: Landmark coordinates as a flat list (18 values) or (9, 2) array.

    Returns:
        224x224 BGR image aligned by eye-to-eye similarity transform.
    """
    if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("Input image must be an HxWx3 BGR array.")

    pts = _ensure_points_array(pts)
    left_eye = pts[0:2].mean(axis=0)
    right_eye = pts[2:4].mean(axis=0)

    # Estimate similarity transform mapping input eye positions to canonical targets.
    src = np.vstack([left_eye, right_eye]).astype(np.float32)
    dst = np.vstack([LEFT_EYE_TARGET, RIGHT_EYE_TARGET]).astype(np.float32)

    transform = cv2.estimateAffinePartial2D(src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2))[0]
    if transform is None:
        raise RuntimeError("Failed to compute similarity transform for landmarks.")

    aligned = cv2.warpAffine(
        img_bgr,
        transform,
        (CANVAS_SIZE, CANVAS_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return aligned


def load_cat_points(cat_txt_path: str) -> np.ndarray:
    """Load nine landmark points from a *.jpg.cat file."""
    with open(cat_txt_path, "r", encoding="utf-8") as f:
        tokens = [float(t) for t in f.read().strip().split()]
    if len(tokens) != NUM_LANDMARKS * 2:
        raise ValueError(f"Expected {NUM_LANDMARKS*2} coordinates in {cat_txt_path}, got {len(tokens)}")
    pts = np.array(tokens, dtype=np.float32).reshape(NUM_LANDMARKS, 2)
    return pts


def _candidate_cat_paths(img_path: str) -> list[str]:
    stem, ext = os.path.splitext(img_path)
    candidates = [f"{stem}.cat"]
    if ext:
        candidates.insert(0, f"{img_path}.cat")
    return candidates


def try_align_from_cat(img_path: str) -> Optional[np.ndarray]:
    """Try to align an image using accompanying CAT landmark files.

    Args:
        img_path: Path to the image file (jpg/png) that may have a .cat landmark file.

    Returns:
        224x224 BGR aligned image if landmarks exist and alignment succeeds, else None.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    for cat_path in _candidate_cat_paths(img_path):
        if os.path.exists(cat_path):
            pts = load_cat_points(cat_path)
            try:
                return align_face_bye2e(img, pts)
            except Exception as exc:  # keep searching other candidates if available
                print(f"WARN: failed to align using {cat_path}: {exc}")
    return None
