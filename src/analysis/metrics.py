"""
Information-Theoretic and Geometric Representation Metrics.

"""

from typing import Dict, Any
import numpy as np


def compute_rankme(features: np.ndarray, epsilon: float = 1e-12) -> float:
    """
    Computes RankMe (Effective Rank) via the entropy of normalized singular values.
    Measures the degree of dimensional collapse in the representation space.

    Formula:
        p_k = \sigma_k / \sum_i \sigma_i
        RankMe = \exp( - \sum_k p_k \log(p_k) )

    Args:
        features: (N, D) array of representations
        epsilon: Small numerical stabilizer to avoid log(0)

    Returns:
        float: Effective rank score
    """
    x = features.astype(np.float64)
    # Center features
    x = x - np.mean(x, axis=0, keepdims=True)

    # Singular Value Decomposition
    s = np.linalg.svd(x, compute_uv=False)
    sum_s = np.sum(s)

    if sum_s < epsilon:
        return 0.0

    p = s / sum_s
    # Filter zeros
    p = p[p > epsilon]
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))

