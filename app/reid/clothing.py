"""
Clothing Feature Extractor & Matcher for CCTV Person Re-ID.

Extracts normalized upper-body (torso) and lower-body (legs) HSV color histograms
from person crops. Because a person's clothing colors remain invariant when they
turn around (front vs back), comparing upper and lower clothing colors provides
robust identity continuity when face and frontal details are hidden.
"""
import cv2
import numpy as np


class ClothingProfile:
    """Stores normalized HSV histograms for upper and lower body clothing."""

    def __init__(self, upper_hist: np.ndarray, lower_hist: np.ndarray, quality: float = 1.0):
        self.upper_hist = upper_hist  # shape (34,), L2-normalized
        self.lower_hist = lower_hist  # shape (34,), L2-normalized
        self.quality    = quality

    def to_dict(self):
        return {
            'upper': self.upper_hist,
            'lower': self.lower_hist,
            'quality': self.quality
        }


def extract_clothing_features(bgr_crop: np.ndarray, bins_h: int = 18, bins_s: int = 8, bins_v: int = 8) -> ClothingProfile:
    """
    Extract upper and lower body clothing color histograms from a person crop.
    
    bgr_crop: BGR numpy image of the person bounding box.
    Returns: ClothingProfile or None if crop is invalid/empty.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return None

    h, w = bgr_crop.shape[:2]
    if h < 30 or w < 15:
        return None

    try:
        hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    except Exception:
        return None

    # Upper Body (Torso): ~15% to 55% of height, central 65% of width to avoid background clutter
    uy1 = int(h * 0.15)
    uy2 = int(h * 0.55)
    ux1 = int(w * 0.18)
    ux2 = int(w * 0.82)
    upper_roi = hsv[uy1:uy2, ux1:ux2]
    if upper_roi.size == 0:
        upper_roi = hsv[: h // 2, :]

    # Lower Body (Legs / Pants): ~55% to 88% of height, central 60% of width
    ly1 = int(h * 0.55)
    ly2 = int(h * 0.88)
    lx1 = int(w * 0.20)
    lx2 = int(w * 0.80)
    lower_roi = hsv[ly1:ly2, lx1:lx2]
    if lower_roi.size == 0:
        lower_roi = hsv[h // 2 :, :]

    def _calc_hist(roi):
        # 1D histograms across H, S, V channels
        h_hist = cv2.calcHist([roi], [0], None, [bins_h], [0, 180]).flatten()
        s_hist = cv2.calcHist([roi], [1], None, [bins_s], [0, 256]).flatten()
        v_hist = cv2.calcHist([roi], [2], None, [bins_v], [0, 256]).flatten()

        # Weighting: Hue is primary color, Saturation is purity, Value is shade (black/white)
        # For low saturation, V is more informative; for high saturation, H is most informative
        # Normalize each channel first
        s_sum = s_hist.sum()
        if s_sum > 0:
            s_hist /= s_sum
        h_sum = h_hist.sum()
        if h_sum > 0:
            h_hist /= h_sum
        v_sum = v_hist.sum()
        if v_sum > 0:
            v_hist /= v_sum

        hist = np.concatenate([h_hist * 0.50, s_hist * 0.25, v_hist * 0.25]).astype(np.float32)
        norm = np.linalg.norm(hist)
        if norm > 1e-6:
            hist /= norm
        return hist

    upper_hist = _calc_hist(upper_roi)
    lower_hist = _calc_hist(lower_roi)

    # Simple quality metric based on resolution
    quality = float(np.clip((h * w) / (80 * 40), 0.2, 1.0))
    return ClothingProfile(upper_hist, lower_hist, quality=quality)


def compare_clothing(cloth1, cloth2, w_upper: float = 0.60, w_lower: float = 0.40) -> float:
    """
    Compute cosine similarity between two clothing profiles.
    Returns float in [0.0, 1.0].
    """
    if cloth1 is None or cloth2 is None:
        return 0.0

    u1 = cloth1.upper_hist if isinstance(cloth1, ClothingProfile) else cloth1.get('upper')
    u2 = cloth2.upper_hist if isinstance(cloth2, ClothingProfile) else cloth2.get('upper')
    l1 = cloth1.lower_hist if isinstance(cloth1, ClothingProfile) else cloth1.get('lower')
    l2 = cloth2.lower_hist if isinstance(cloth2, ClothingProfile) else cloth2.get('lower')

    if u1 is None or u2 is None or l1 is None or l2 is None:
        return 0.0

    # Cosine similarities
    sim_upper = float(np.clip(np.dot(u1, u2), 0.0, 1.0))
    sim_lower = float(np.clip(np.dot(l1, l2), 0.0, 1.0))

    return float(np.clip(w_upper * sim_upper + w_lower * sim_lower, 0.0, 1.0))
