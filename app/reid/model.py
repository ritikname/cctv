"""
Re-ID Feature Extractor

Uses ResNet50 (much better than ResNet18) with multi-layer feature
concatenation for richer person representations.
Also adds color histogram features as a complementary signal.
"""
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import warnings

try:
    from torchreid.utils import FeatureExtractor
    HAS_TORCHREID = True
except ImportError:
    HAS_TORCHREID = False

from app.config import Config
from app.reid.clothing import extract_clothing_features


class ReIDModel:
    def __init__(self):
        self.device = Config.DEVICE
        self.model_name = Config.REID_MODEL

        if HAS_TORCHREID:
            self.extractor = FeatureExtractor(
                model_name=self.model_name,
                model_path='',
                device=self.device
            )
            self._mode = "torchreid"
        else:
            import torchvision.models as models
            import torchvision.transforms as T

            # ResNet50 gives 2048-dim features (much richer than ResNet18's 512)
            self.model = models.resnet50(weights="DEFAULT").to(self.device)
            self.model.fc = nn.Identity()
            self.model.eval()

            self.transforms = T.Compose([
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
            self._mode = "fallback"

    # ── colour histogram (complementary to CNN) ──────────────────────────────
    @staticmethod
    def _colour_histogram(bgr_crop, bins=16):
        """
        Compute a normalised colour histogram in HSV space.
        Returns a 1-D vector of length bins*3.
        Lightweight, captures dominant clothing colours.
        """
        hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        # Use the central 60% to avoid background
        cy, cx = h // 2, w // 2
        rh, rw = max(h // 5, 1), max(w // 5, 1)
        roi = hsv[cy - rh:cy + rh, cx - rw:cx + rw]
        if roi.size == 0:
            roi = hsv

        hists = []
        for ch in range(3):
            hist = cv2.calcHist([roi], [ch], None, [bins], [0, 256])
            hist = hist.flatten()
            s = hist.sum()
            if s > 0:
                hist /= s
            hists.append(hist)
        return np.concatenate(hists).astype(np.float32)

    def extract_clothing(self, crops):
        """Extract ClothingProfile for each BGR crop."""
        return [extract_clothing_features(c) for c in crops]

    # ── main extraction ──────────────────────────────────────────────────────
    def extract_features(self, crops):
        """
        Extract Re-ID features from a list of BGR crops (numpy arrays).
        Returns numpy array of shape (N, D).
        """
        if len(crops) == 0:
            return np.array([])

        if self._mode == "torchreid":
            rgb_crops = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]
            with torch.no_grad():
                features = self.extractor(rgb_crops).cpu().numpy()
            return features

        # ── Fallback: ResNet50 features + colour histogram ──────────────
        # CRITICAL: ResNet50 uses ReLU activations, producing all-positive
        # embeddings. When L2-normalised, ALL cross-person cosine similarities
        # cluster around 0.72-0.88, making the 0.75 threshold completely
        # useless for discrimination. We must whiten (zero-centre) the CNN
        # embeddings BEFORE L2 normalisation so cosine similarity is
        # discriminative (negatives are possible for different people).
        rgb_crops = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]
        pil_crops = [Image.fromarray(c) for c in rgb_crops]
        batch = torch.stack([self.transforms(c) for c in pil_crops]).to(self.device)

        with torch.no_grad():
            cnn_feats = self.model(batch).cpu().numpy()  # raw, un-normalised

        # Whiten: subtract per-dimension mean across batch so features are
        # zero-centred. For a single crop, we subtract the feature's own mean.
        if cnn_feats.shape[0] > 1:
            # Batch whitening: subtract batch mean per dimension
            cnn_feats = cnn_feats - cnn_feats.mean(axis=0, keepdims=True)
        else:
            # Single-sample: subtract the feature vector's own mean
            cnn_feats = cnn_feats - cnn_feats.mean(axis=1, keepdims=True)

        # L2 normalise after whitening
        norms = np.linalg.norm(cnn_feats, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1e-6
        cnn_feats = cnn_feats / norms

        # Append colour histograms (higher weight now that CNN is whitened)
        colour_feats = np.array([self._colour_histogram(c) for c in crops])
        norms = np.linalg.norm(colour_feats, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1e-6
        colour_feats = colour_feats / norms

        # Weight: 65% whitened CNN + 35% colour (colour is more discriminative
        # in fallback mode because the CNN is not trained for Re-ID)
        combined = np.concatenate([cnn_feats * 0.65, colour_feats * 0.35], axis=1)
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1e-6
        return (combined / norms).astype(np.float32)
