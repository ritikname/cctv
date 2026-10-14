"""
Multi-View Appearance & Clothing Gallery for CCTV Person Re-ID.

Stores partitioned viewpoints (front, back, side, general) and clothing profiles
(upper & lower HSV histograms) per person.
Prevents viewpoint collapse: front views are never evicted when a person turns
and shows their back, and back views are captured and retained for re-identifying
retreating or exiting visitors.
"""
from collections import deque
import numpy as np
from app.config import Config
from app.reid.clothing import ClothingProfile, compare_clothing


class MultiViewGallery:
    """
    Multi-view appearance and clothing gallery for one person.
    Partitions embeddings by observed viewpoint (front, back, side, general)
    and maintains rolling upper/lower body clothing profiles.
    """

    def __init__(self, max_per_view: int = 4, max_general: int = 8):
        self.max_per_view = max_per_view
        self._views = {
            'front': deque(maxlen=max_per_view),
            'back': deque(maxlen=max_per_view),
            'side': deque(maxlen=max_per_view),
            'general': deque(maxlen=max_general),
        }
        self.clothing_profiles: deque = deque(maxlen=6)
        self.mean_upper = None
        self.mean_lower = None

    # ── write ────────────────────────────────────────────────────────────────
    def add(self, embedding: np.ndarray, view_tag: str = 'general', clothing: ClothingProfile = None) -> None:
        """
        Normalise and store embedding in the appropriate viewpoint partition.
        Optionally records clothing profile.
        """
        if embedding is not None:
            norm = np.linalg.norm(embedding)
            if norm > 1e-6:
                normalized = embedding / norm
                if view_tag not in self._views:
                    view_tag = 'general'
                self._views[view_tag].append(normalized)

        if clothing is not None:
            self.add_clothing(clothing)

    def add_clothing(self, clothing: ClothingProfile) -> None:
        """Store clothing profile and update mean upper/lower histograms."""
        if clothing is None:
            return
        self.clothing_profiles.append(clothing)

        # Update mean upper and lower histograms
        uppers = [p.upper_hist for p in self.clothing_profiles if p.upper_hist is not None]
        lowers = [p.lower_hist for p in self.clothing_profiles if p.lower_hist is not None]

        if uppers:
            mean_u = np.mean(uppers, axis=0)
            nu = np.linalg.norm(mean_u)
            self.mean_upper = (mean_u / nu).astype(np.float32) if nu > 1e-6 else mean_u

        if lowers:
            mean_l = np.mean(lowers, axis=0)
            nl = np.linalg.norm(mean_l)
            self.mean_lower = (mean_l / nl).astype(np.float32) if nl > 1e-6 else mean_l

    # ── read ─────────────────────────────────────────────────────────────────
    def is_empty(self) -> bool:
        return sum(len(dq) for dq in self._views.values()) == 0

    def size(self) -> int:
        return sum(len(dq) for dq in self._views.values())

    def view_counts(self) -> dict:
        return {k: len(v) for k, v in self._views.items()}

    def as_matrix(self) -> np.ndarray:
        """Return (N, D) matrix of all stored embeddings across all viewpoints."""
        all_embeddings = [e for dq in self._views.values() for e in dq]
        if not all_embeddings:
            return np.empty((0, 512), dtype=np.float32)
        return np.stack(all_embeddings, axis=0)

    # ── matching ─────────────────────────────────────────────────────────────
    def best_similarity(self, query: np.ndarray, view_tag: str = None) -> float:
        """
        Compare query embedding against stored gallery embeddings.
        Returns the HIGHEST cosine similarity across all viewpoints (or specific view if specified).
        """
        if self.is_empty() or query is None:
            return 0.0

        q_norm = np.linalg.norm(query)
        if q_norm < 1e-6:
            return 0.0
        q = query / q_norm

        # If a specific view is requested and present, test that view first
        if view_tag and view_tag in self._views and len(self._views[view_tag]) > 0:
            view_mat = np.stack(list(self._views[view_tag]), axis=0)
            view_sim = float(np.max(view_mat @ q))
            gallery_mat = self.as_matrix()
            overall_sim = float(np.max(gallery_mat @ q))
            return max(view_sim, overall_sim)

        gallery_mat = self.as_matrix()
        sims = gallery_mat @ q
        return float(np.max(sims))

    def clothing_similarity(self, query_clothing: ClothingProfile) -> float:
        """
        Compare query clothing against stored clothing profile.
        Returns max similarity across exemplars or comparison to mean profile.
        """
        if query_clothing is None or not self.clothing_profiles:
            return 0.0

        # Compare against mean profile if available
        sim_mean = 0.0
        if self.mean_upper is not None and self.mean_lower is not None:
            mean_prof = ClothingProfile(self.mean_upper, self.mean_lower)
            sim_mean = compare_clothing(query_clothing, mean_prof)

        # Compare against all stored exemplars and take max
        sim_max = max(compare_clothing(query_clothing, p) for p in self.clothing_profiles)

        return float(max(sim_mean, sim_max))

    def mean_similarity(self, query: np.ndarray) -> float:
        if self.is_empty() or query is None:
            return 0.0
        q = query / max(np.linalg.norm(query), 1e-6)
        return float(np.mean(self.as_matrix() @ q))


# Backward compatibility alias
RollingGallery = MultiViewGallery
