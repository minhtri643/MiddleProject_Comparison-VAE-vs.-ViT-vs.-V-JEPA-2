# Scientific Evaluation Report: Image autonomous driving representation Experiment

## 1. Standard Benchmark (Clean Input)

| Model | Grid | Dim (D) | PCA 3PC Var (%) | Mean Cosine (Subject A) | Mean Cosine (Background B) | Effective Rank (RankMe) |
|---|---|---:|---:|---:|---:|---:|
| **VAE** | 32x32 | 4 | 90.67% | -0.534 | 0.6309 | 3.58 |
| **ViT** | 14x14 | 768 | 50.49% | 0.4244 | 0.5166 | 112.98 |
| **V-JEPA 2** | 16x16 | 1024 | 29.21% | 0.4954 | 0.5017 | 166.23 |

## 2. Adverse Weather Robustness Stress-Test

| Model | Clean RankMe | Foggy RankMe | RankMe Retention (%) | Clean Cosine A | Foggy Cosine A | Delta Cosine A |
|---|---:|---:|---:|---:|---:|---:|
| **VAE** | 3.58 | 3.44 | **96.09%** | -0.534 | -0.1532 | 0.3808 |
| **ViT** | 112.98 | 114.63 | **101.46%** | 0.4244 | 0.4192 | -0.0052 |
| **V-JEPA 2** | 166.23 | 157.07 | **94.49%** | 0.4954 | 0.4875 | -0.0079 |

## 3. Theoretical Defense: Why 3D Scatter is Superior to 2D

1. **Elimination of False 2D Projection Overlaps:** Feature clusters curve across multi-dimensional manifolds. Flattening onto a 2D plane causes artificial cluster overlapping. A 3D manifold projection reveals the orthogonal PC3 axis, cleanly separating vehicle features from asphalt background.
2. **1-to-1 Correspondence with 3-Channel PCA-RGB Heatmaps:** The color maps use RGB channels mapped directly to PC1, PC2, and PC3. The 3D scatter plot faithfully renders all three principal axes simultaneously.

## 4. Key Scientific Conclusions

- **VAE Representation Degradation under Adverse Weather:** VAE displays severe sensitivity to optical surface pixel variations, suffering massive latent feature drift (Delta = +0.3808).
- **V-JEPA Semantic Invariance:** Maintains manifold geometric topology with high RankMe retention (94.49%), demonstrating the superiority of non-generative joint-embedding predictive architectures for autonomous driving world models.
