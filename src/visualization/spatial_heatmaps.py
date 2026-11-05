"""
Spatial Representation & Similarity Heatmap Visualizations.
Produces publication-quality multi-panel comparative figures.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_spatial_comparison(
    display_image: np.ndarray,
    results: Dict[str, Dict[str, Any]],
    queries: Dict[str, Tuple[float, float]],
    save_path: Path
):
    """
    Renders a comprehensive 4-row or 3-row comparison figure:
    Columns: [Original / PCA RGB], [Cosine Sim Query A], [Cosine Sim Query B]
    Rows: VAE, ViT, V-JEPA 2
    """
    models = list(results.keys())
    n_models = len(models)

    fig, axes = plt.subplots(n_models, 3, figsize=(15, 4.5 * n_models), constrained_layout=True)
    if n_models == 1:
        axes = np.expand_dims(axes, 0)

    for i, model_name in enumerate(models):
        res = results[model_name]
        rgb_map = res["pca_rgb"]
        var_pct = res["variance_explained"]
        cos_a = res["cosine_maps"]["A_subject"]
        cos_b = res["cosine_maps"]["B_background"]

        # Column 1: PCA RGB Map
        ax0 = axes[i, 0]
        ax0.imshow(rgb_map)
        ax0.set_title(f"{model_name}\nPCA 3-Channel RGB (Var: {var_pct:.1f}%)", fontsize=11, fontweight="bold")
        ax0.set_ylabel(f"{model_name}\nGrid: {res['grid_shape'][:2]}", fontsize=10, fontweight="bold")
        ax0.set_xticks([])
        ax0.set_yticks([])

        # Column 2: Cosine Similarity with Subject (Query A)
        ax1 = axes[i, 1]
        im1 = ax1.imshow(cos_a, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        qa = queries["A_subject"]
        qa_row, qa_col = res["query_indices"]["A_subject"]
        ax1.plot(qa_col, qa_row, "kX", markersize=10, markeredgewidth=2, label="Query A (Subject)")
        ax1.set_title(f"{model_name} - Cosine Sim vs A\n(Mean excl. self: {res['mean_cosine']['A_subject']:.3f})", fontsize=10)
        ax1.set_xticks([])
        ax1.set_yticks([])
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # Column 3: Cosine Similarity with Background (Query B)
        ax2 = axes[i, 2]
        im2 = ax2.imshow(cos_b, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        qb_row, qb_col = res["query_indices"]["B_background"]
        ax2.plot(qb_col, qb_row, "gX", markersize=10, markeredgewidth=2, label="Query B (Background)")
        ax2.set_title(f"{model_name} - Cosine Sim vs B\n(Mean excl. self: {res['mean_cosine']['B_background']:.3f})", fontsize=10)
        ax2.set_xticks([])
        ax2.set_yticks([])
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

