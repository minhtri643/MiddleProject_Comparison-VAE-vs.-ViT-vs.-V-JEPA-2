# Phase 1: Spatial Manifold Topology & Representation Extraction

## 1. Experimental Overview
- **Objective:** Extract the latent state representation ($S_t$) across the three models on Canonical Keyframe 56.
- **Methods:**
  - 3-Channel PCA-RGB projection mapping principal axes PC1, PC2, PC3 to red, green, and blue.
  - Local patch Cosine similarity mapping relative to Query A (Vehicle foreground, $[0.50, 0.65]$) and Query B (Asphalt roadway background, $[0.10, 0.12]$).
  - 2D PCA patch embedding scatter visualization.

## 2. Generated Visual Artifacts
1. `01_spatial_comparison.png`: Comprehensive comparative multi-panel figure displaying PCA-RGB maps and Cosine similarity heatmaps.
2. `02_patch_scatter_2d.png`: 2D PCA projection scatter plot colored by Cosine similarity to Query A.

## 3. Key Observations
- VAE concentrates 90.67% of its total variance into the first 3 principal components, producing visually sharp but semantically constrained 2D clusters.
- V-JEPA 2 distributes its representational energy uniformly across 1024 dimensions (only 29.21% in top-3 PCs), reflecting a high-capacity semantic state space.
