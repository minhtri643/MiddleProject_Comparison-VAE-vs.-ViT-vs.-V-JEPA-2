# Phase 5: Target State Forecasting & Temporal Causality Stress-Testing

## 1. Experimental Overview
- **Objective:** Test whether representations internalize temporal dynamics and causality by forecasting future Target Frame 56 ($S_{56}$) under chronological vs randomly scrambled sequence conditions.
- **Setup:**
  - Sequence: 16 consecutive frames ($t = 0 \to 15$, Frames 41 to 56).
  - Target Frame: Frame 56 ($t = 15$).
  - Context Sequence: Preceding 15 frames ($t = 0 \to 14$).
- **Scenarios:**
  - **Scenario A (Chronological Flow):** Natural forward arrow of time ($t_0 \to t_1 \to \dots \to t_{14}$).
  - **Scenario B (Shuffled Context Flow):** Context frames randomly permuted via permutation operator $\pi$.

## 2. Generated Visual Artifacts & Metrics
1. `06_temporal_target_prediction.png`: Dual bar chart comparing Cosine Similarity and MSE under chronological vs shuffled conditions.
2. `07_temporal_trajectory_pca.png`: 2D PCA temporal manifold projection comparing smooth directed trajectories against shattered chaotic paths.
3. `06_temporal_causality_metrics.csv`: Tabular quantitative measurements of temporal prediction accuracy and degradation.

## 3. Quantitative Summary
| Model | Chronological CosSim | Shuffled CosSim | Cosine Drop ($\Delta_{\text{Shuffle}}$) | Chrono MSE | Shuffled MSE | MSE Jump Factor | Scientific Insight |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **SD-VAE** | **0.9996** | 0.9367 | 6.29% | 0.0152 | 1.0948 | 72.0$\times$ | High-frequency surface pixel coding; distribution drift under scrambling. |
| **ViT-Base** | 0.9615 | 0.8435 | **11.80%** | 0.0323 | 0.1300 | 4.0$\times$ | Sensitive to sequence order; loses semantic manifold continuity. |
| **Meta V-JEPA 2** | 0.9833 | 0.8820 | **10.13%** | 0.1039 | **0.7516** | **7.23$\times$** | **Internalizes physical momentum and the Arrow of Time; MSE jumps 7.23-fold.** |
