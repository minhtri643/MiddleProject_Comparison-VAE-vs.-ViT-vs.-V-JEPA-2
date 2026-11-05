"""
Principal Component Analysis (PCA) for High-Dimensional Patch Embeddings.

"""

from typing import Tuple, Dict, Any
import numpy as np
from sklearn.decomposition import PCA


def compute_patch_pca(
    grid: np.ndarray,
    n_components: int = 3,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, float, PCA]:
    """
    Applies PCA on flattened patch features of shape (H, W, D).

    Args:
        grid: np.ndarray of shape (H, W, D)
        n_components: Number of components (default 3 for RGB mapping)
        normalize: Whether to L2-normalize vectors prior to PCA

    Returns:
        rgb_map: (H, W, 3) mapped into [0, 1] for visual display
        reduced_vectors: (N, n_components) projected vectors
        variance_explained_ratio: Total variance explained by the top n_components
        pca_model: Fitted sklearn PCA object
    """
    h, w, d = grid.shape
    features = grid.reshape(-1, d).astype(np.float32)

    if normalize:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        features = features / norms

    pca = PCA(n_components=n_components, random_state=42)
    projected = pca.fit_transform(features)  # (N, n_components)

    variance_ratio = float(np.sum(pca.explained_variance_ratio_) * 100.0)

    # Scale projected components to [0, 1] per channel for RGB rendering
    rgb_flat = np.zeros_like(projected[:, :3])
    for i in range(min(3, n_components)):
        col = projected[:, i]
        min_v, max_v = col.min(), col.max()
        if max_v - min_v > 1e-7:
            rgb_flat[:, i] = (col - min_v) / (max_v - min_v)
        else:
            rgb_flat[:, i] = 0.5

    rgb_map = rgb_flat.reshape(h, w, 3)
    return rgb_map, projected, variance_ratio, pca

