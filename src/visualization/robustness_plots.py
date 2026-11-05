"""
Robustness Stress-Test Visualizations under Adverse Weather.
"""

from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np


def plot_robustness_stress_test(
    clean_display: np.ndarray,
    foggy_display: np.ndarray,
    clean_results: Dict[str, Dict[str, Any]],
    foggy_results: Dict[str, Dict[str, Any]],
    save_path: Path
):
    """
    Renders side-by-side comparison under Clean vs Foggy Adverse Weather conditions.
    Row 1: [Clean Scene], [VAE Clean PCA], [ViT Clean PCA], [V-JEPA Clean PCA]
    Row 2: [Foggy Scene], [VAE Foggy PCA], [ViT Foggy PCA], [V-JEPA Foggy PCA]
    """
    models = ["VAE", "ViT", "V-JEPA 2"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)

    # Row 1: Clean
    axes[0, 0].imshow(clean_display)
    axes[0, 0].set_title("1. Clean Input (Ideal Weather)", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    for col, m in enumerate(models, start=1):
        res = clean_results[m]
        axes[0, col].imshow(res["pca_rgb"])
        axes[0, col].set_title(f"{m} (Clean)\nRankMe: {res['effective_rank']:.1f}", fontsize=11, fontweight="bold")
        axes[0, col].axis("off")

    # Row 2: Corrupted (Fog / Adverse Weather)
    axes[1, 0].imshow(foggy_display)
    axes[1, 0].set_title("2. Adverse Weather (Dense Fog & Haze)", fontsize=11, fontweight="bold", color="darkred")
    axes[1, 0].axis("off")

    for col, m in enumerate(models, start=1):
        res = foggy_results[m]
        axes[1, col].imshow(res["pca_rgb"])
        axes[1, col].set_title(f"{m} (Foggy)\nRankMe: {res['effective_rank']:.1f}", fontsize=11, fontweight="bold", color="darkred")
        axes[1, col].axis("off")

    plt.suptitle("ADVERSE WEATHER ROBUSTNESS STRESS-TEST", fontsize=14, fontweight="bold")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

