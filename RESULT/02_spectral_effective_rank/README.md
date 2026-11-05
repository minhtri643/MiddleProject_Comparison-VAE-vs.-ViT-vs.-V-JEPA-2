# Phase 2: Spectral Dimensionality & Effective Rank (RankMe)

## 1. Experimental Overview
- **Objective:** Overcome false 2D projection overlaps through orthogonal 3D PCA manifold projection and rigorously quantify dimensional collapse via **RankMe**.
- **Theoretical Formulation:**
  $$\text{RankMe}(\mathbf{Z}) = \exp\left(-\sum_{k=1}^{\min(N, D)} p_k \ln p_k\right), \quad p_k = \frac{\sigma_k}{\sum_{i} \sigma_i}$$
  where $\sigma_k$ represents the singular values obtained from Singular Value Decomposition (SVD) of the centered patch representation matrix $\mathbf{Z}$.

## 2. Generated Visual Artifacts & Metrics
1. `02_patch_scatter_3d.png`: 3D manifold projection in orthogonal PCA space.
2. `03_quantitative_metrics.csv`: Master quantitative metrics recording Grid size, Dimension $D$, Top-3 PC Variance, Cosine margins, and RankMe scores.
3. `04_research_report.md`: Automated technical evaluation report.

## 3. Quantitative Summary
- **SD-VAE:** RankMe = **3.58** / 4.0 (Constrained by $D=4$).
- **ViT-Base:** RankMe = **112.98** / 195.0.
- **Meta V-JEPA 2:** RankMe = **166.23** / 255.0 (Highest effective dimensionality, eliminating dimensional collapse).
