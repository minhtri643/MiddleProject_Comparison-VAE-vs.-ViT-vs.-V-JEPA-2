"""
ViT Linear Patch Decoder (MAE-style Generative Readout).

  Masked Autoencoders Are Scalable Vision Learners. Section 3.2.
"""

from typing import Optional, Tuple
import numpy as np
from sklearn.linear_model import Ridge
from .patch_utils import patches_to_image, resize_image


class ViTDecoder:
    """
    Decodes ViT spatial patch tokens in R^(14 x 14 x 768) to pixel space R^(224 x 224 x 3)
    (and resizes to 256x256 for unified benchmark comparison) using an MAE-style
    linear projection probe.
    """
    def __init__(self, alpha: float = 1.0, patch_size: int = 16):
        self.alpha = alpha
        self.patch_size = patch_size
        self.probe = Ridge(alpha=self.alpha, fit_intercept=True)
        self.is_fitted = False

    def fit(self, tokens_flat: np.ndarray, patches_flat: np.ndarray):
        """
        Fit the optimal linear decoder W* mapping latent tokens to pixel patches:
        min_W ||Tokens @ W + b - Patches||^2 + alpha * ||W||_F^2

        Args:
            tokens_flat: np.ndarray of shape (N_samples, 768)
            patches_flat: np.ndarray of shape (N_samples, 768)
        """
        print(f"[ViTDecoder] Fitting MAE linear probe on {tokens_flat.shape[0]} patch pairs (D=768)...")
        self.probe.fit(tokens_flat, patches_flat)
        self.is_fitted = True
        print("[ViTDecoder] Fit completed successfully.")

    def decode(self, token_grid: np.ndarray, target_shape: Tuple[int, int] = (256, 256)) -> np.ndarray:
        """
        Decode a (14, 14, 768) token grid back to image space.

        Args:
            token_grid: np.ndarray of shape (14, 14, 768)
            target_shape: final output dimensions (H, W), default (256, 256)

        Returns:
            reconstructed_image: np.ndarray of shape (target_shape[0], target_shape[1], 3) uint8
        """
        if not self.is_fitted:
            raise RuntimeError("ViTDecoder must be fitted with training patch representations before decode().")

        if token_grid.shape != (14, 14, 768):
            raise ValueError(f"Expected ViT token grid of shape (14, 14, 768), got {token_grid.shape}")

        tokens_flat = token_grid.reshape(14 * 14, 768)
        decoded_patches = self.probe.predict(tokens_flat)

        # Reassemble to 224x224 (14 * 16 = 224)
        img_224 = patches_to_image(decoded_patches, target_h=224, target_w=224, patch_size=self.patch_size)

        # Scale to target shape if requested (e.g. 256x256 to align with VAE & V-JEPA)
        if target_shape == (224, 224):
            return img_224
        else:
            return resize_image(img_224, target_shape)
