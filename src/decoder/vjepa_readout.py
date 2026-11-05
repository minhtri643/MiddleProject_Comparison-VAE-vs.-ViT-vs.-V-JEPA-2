"""
V-JEPA Generative Readout Probe.

  Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. Section 4.3 "Generative Readout".
"""

from typing import Tuple
import numpy as np
from sklearn.linear_model import Ridge
from .patch_utils import patches_to_image, resize_image


class VJEPAReadout:
    """
    Decodes V-JEPA 2 continuous spatial patch tokens in R^(16 x 16 x 1024) to pixel space
    R^(256 x 256 x 3) using a regularized generative readout probe.
    """
    def __init__(self, alpha: float = 1.0, patch_size: int = 16):
        self.alpha = alpha
        self.patch_size = patch_size
        self.probe = Ridge(alpha=self.alpha, fit_intercept=True)
        self.is_fitted = False

    def fit(self, tokens_flat: np.ndarray, patches_flat: np.ndarray):
        """
        Fit the optimal linear generative readout mapping V-JEPA tokens to pixel patches:
        min_W ||Tokens @ W + b - Patches||^2 + alpha * ||W||_F^2

        Args:
            tokens_flat: np.ndarray of shape (N_samples, 1024)
            patches_flat: np.ndarray of shape (N_samples, 768)
        """
        print(f"[VJEPAReadout] Fitting generative readout on {tokens_flat.shape[0]} patch pairs (D=1024 -> 768)...")
        self.probe.fit(tokens_flat, patches_flat)
        self.is_fitted = True
        print("[VJEPAReadout] Fit completed successfully.")

    def decode(self, token_grid: np.ndarray, target_shape: Tuple[int, int] = (256, 256)) -> np.ndarray:
        """
        Decode a (16, 16, 1024) token grid back to image space (256, 256, 3).

        Args:
            token_grid: np.ndarray of shape (16, 16, 1024)
            target_shape: final output dimensions (H, W), default (256, 256)

        Returns:
            reconstructed_image: np.ndarray of shape (256, 256, 3) uint8
        """
        if not self.is_fitted:
            raise RuntimeError("VJEPAReadout must be fitted with training patch representations before decode().")

        if token_grid.shape != (16, 16, 1024):
            raise ValueError(f"Expected V-JEPA token grid of shape (16, 16, 1024), got {token_grid.shape}")

        tokens_flat = token_grid.reshape(16 * 16, 1024)
        decoded_patches = self.probe.predict(tokens_flat)

        # Reassemble to 256x256 (16 * 16 = 256)
        img_256 = patches_to_image(decoded_patches, target_h=256, target_w=256, patch_size=self.patch_size)

        if target_shape == (256, 256):
            return img_256
        else:
            return resize_image(img_256, target_shape)
