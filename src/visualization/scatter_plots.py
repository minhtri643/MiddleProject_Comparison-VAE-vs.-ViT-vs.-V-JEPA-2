"""
2D and 3D PCA Patch Embedding Scatter Plot Visualizations.
"""

from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_patch_scatter(results: Dict[str, Dict[str, Any]], save_path: Path):
    """
    Renders 2D PCA patch projection side-by-side for all evaluated models.
    Points represent individual spatial patches, colored by Cosine similarity to Query A.
    """
    models = list(results.keys())
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5.5), constrained_layout=True)
    if n_models == 1:
        axes = [axes]

    for i, model_name in enumerate(models):
        res = results[model_name]
        projected = res["projected_pca"]  # (N, >=2)
        h, w = res["grid_shape"][:2]
        qa_row, qa_col = res["query_indices"]["A_subject"]
        qb_row, qb_col = res["query_indices"]["B_background"]

        qa_flat = qa_row * w + qa_col
        qb_flat = qb_row * w + qb_col

        sims_a = res["cosine_maps"]["A_subject"].reshape(-1)

        ax = axes[i]
        scatter = ax.scatter(
            projected[:, 0],
            projected[:, 1],
            c=sims_a,
            cmap="coolwarm",
            alpha=0.8,
            s=35,
            edgecolor="none"
        )

        # Highlight Query Points
        ax.scatter(projected[qa_flat, 0], projected[qa_flat, 1], c="black", marker="X", s=140, label="Query A (Subject)")
        ax.scatter(projected[qb_flat, 0], projected[qb_flat, 1], c="green", marker="X", s=140, label="Query B (Background)")

        ax.set_title(f"{model_name} (2D PCA)\n(RankMe = {res['effective_rank']:.1f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Principal Component 1", fontsize=10)
        ax.set_ylabel("Principal Component 2", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=9)
        fig.colorbar(scatter, ax=ax, label="Cosine Sim to Query A", fraction=0.046, pad=0.04)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_patch_scatter_3d(results: Dict[str, Dict[str, Any]], save_path: Path):
    """
    Renders 3D PCA patch projection (PC1, PC2, PC3) in manifold space.
    Resolves false 2D cluster overlaps and mirrors 3-channel RGB geometric layout.
    """
    models = list(results.keys())
    n_models = len(models)

    fig = plt.figure(figsize=(7 * n_models, 6.5), constrained_layout=True)

    for i, model_name in enumerate(models):
        res = results[model_name]
        projected = res["projected_pca"]  # (N, >=3)
        h, w = res["grid_shape"][:2]
        qa_row, qa_col = res["query_indices"]["A_subject"]
        qb_row, qb_col = res["query_indices"]["B_background"]

        qa_flat = qa_row * w + qa_col
        qb_flat = qb_row * w + qb_col

        sims_a = res["cosine_maps"]["A_subject"].reshape(-1)

        ax = fig.add_subplot(1, n_models, i + 1, projection="3d")
        scatter = ax.scatter(
            projected[:, 0],
            projected[:, 1],
            projected[:, 2],
            c=sims_a,
            cmap="coolwarm",
            alpha=0.85,
            s=40,
            edgecolors="none"
        )

        # Highlight Query Points in 3D
        ax.scatter(projected[qa_flat, 0], projected[qa_flat, 1], projected[qa_flat, 2],
                   c="black", marker="X", s=180, depthshade=False, label="Query A (Subject)")
        ax.scatter(projected[qb_flat, 0], projected[qb_flat, 1], projected[qb_flat, 2],
                   c="green", marker="X", s=180, depthshade=False, label="Query B (Background)")

        ax.set_title(f"{model_name} (3D Manifold PCA)\n(RankMe = {res['effective_rank']:.1f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("PC 1", fontsize=9)
        ax.set_ylabel("PC 2", fontsize=9)
        ax.set_zlabel("PC 3", fontsize=9)
        ax.view_init(elev=25, azim=45)
        ax.legend(loc="upper left", fontsize=8)
        fig.colorbar(scatter, ax=ax, label="Cosine Sim to A", fraction=0.03, pad=0.08)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
