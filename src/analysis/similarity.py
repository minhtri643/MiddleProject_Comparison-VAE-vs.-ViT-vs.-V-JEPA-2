"""
Cosine Similarity Analysis on Spatial Patch Representations.

"""

from typing import Tuple
import numpy as np


def location_to_patch_index(grid_shape: Tuple[int, int], location_xy: Tuple[float, float]) -> Tuple[int, int, int]:
    """
    Maps normalized (x, y) coordinates in [0, 1] to discrete grid indices.

    Args:
        grid_shape: (H, W)
        location_xy: (x, y) where x is horizontal [0, 1], y is vertical [0, 1]

    Returns:
        (row, col, flat_index)
    """
    h, w = grid_shape
    x, y = location_xy
    col = min(w - 1, max(0, int(round(x * (w - 1)))))
    row = min(h - 1, max(0, int(round(y * (h - 1)))))
    flat_idx = row * w + col
    return row, col, flat_idx


def compute_cosine_map(
    grid: np.ndarray,
    query_xy: Tuple[float, float]
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Computes cosine similarity between a selected query patch and all other patches.

    Args:
        grid: (H, W, D) feature array
        query_xy: Normalized (x, y) coordinates of the reference point

    Returns:
        cosine_map: (H, W) array of similarity values [-1, 1]
        mean_cosine_excl_self: Average cosine similarity over all other patches
        (row, col): Discrete grid coordinates of the query patch
    """
    h, w, d = grid.shape
    features = grid.reshape(-1, d).astype(np.float32)

    # L2 normalize
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed_features = features / norms

    row, col, flat_idx = location_to_patch_index((h, w), query_xy)
    query_vec = normed_features[flat_idx]

    # Dot product between normalized vectors equals cosine similarity
    sims = np.dot(normed_features, query_vec)  # (N,)

    # Mean similarity excluding self
    mask = np.ones(len(sims), dtype=bool)
    mask[flat_idx] = False
    mean_cosine_excl_self = float(np.mean(sims[mask])) if len(sims) > 1 else 0.0

    cosine_map = sims.reshape(h, w)
    return cosine_map, mean_cosine_excl_self, (row, col)

