# Phase 4: Latent Probing & The Pixel Optimization Trap

## 1. Experimental Overview
- **Objective:** Probe spatial visual information preserved in latent state space ($S_t$) by decoding back to pixel space ($S_t \to \hat{X}$).
- **Decoding Mechanisms:**
  1. **VAE (Native Continuous Deconvolution):** Symmetric convolution-attention decoder native to AutoencoderKL.
  2. **ViT (MAE-Style Linear Patch Probe):** L2-regularized linear Ridge probe ($\alpha = 10.0$) mapping 768-d token to $16 \times 16 \times 3 = 768$ RGB patch pixels.
  3. **V-JEPA 2 (Generative Readout Probe):** Linear Ridge probe ($\alpha = 10.0$) mapping 1024-d spatiotemporal token to 768 RGB patch pixels.
- **Probe Calibration Protocol:** Linear probes calibrated on 12 distributed driving keyframes; evaluation performed on Canonical Keyframe 56.

## 2. Generated Visual Artifacts & Metrics
1. `01_vae_reconstruction.png`: Native VAE reconstruction output.
2. `02_vit_reconstruction.png`: ViT linear patch readout output.
3. `03_vjepa_readout.png`: V-JEPA 2 generative readout output.
4. `04_side_by_side_comparison.png`: 4-panel side-by-side reconstruction benchmark.
5. `05_reconstruction_error_heatmaps.png`: Absolute reconstruction error heatmaps $|I - \hat{I}|$.
6. `05_reconstruction_metrics.csv`: Quantitative comparison of PSNR, SSIM, MSE, and MAE.

## 3. Quantitative Summary
- **SD-VAE:** PSNR = **35.25 dB**, SSIM = **0.9743** (Pixel-level memorization).
- **ViT-Base:** PSNR = 17.37 dB, SSIM = 0.3639 (Geometry and vehicle position preserved).
- **Meta V-JEPA 2:** PSNR = 15.76 dB, SSIM = 0.2451 (Semantic invariance; discards high-frequency pixel noise).
