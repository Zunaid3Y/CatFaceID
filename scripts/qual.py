import cv2
import numpy as np

def pass_quality(bgr, lap_thresh=100.0, min_side=120):
    if bgr is None: return False
    h, w = bgr.shape[:2]
    if min(h,w) < min_side: return False
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lapv = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lapv >= lap_thresh
