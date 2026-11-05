"""
End-to-end Latent Reconstruction and Generative Readout Pipeline.

Scientific Context:
- Evaluates whether latent features S_t can reconstruct the original driving visual scene.
"""

import os
import sys

# Critical for Windows PyTorch + OpenCV / Scikit-Image OpenMP coexistence
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# PyTorch must be imported before matplotlib / cv2 on Windows to initialize CUDA & OpenMP runtime safely
import torch

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("[run_decoder] Booting script...", flush=True)

try:
    import argparse
    import traceback
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    print("[run_decoder] Standard libraries imported", flush=True)

    # Setup paths for Middle_Project
    SCRIPTS_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPTS_DIR.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    print(f"[run_decoder] Paths configured: PROJECT_ROOT={PROJECT_ROOT}", flush=True)

    from src.decoder.patch_utils import resize_image
    print("[run_decoder] patch_utils imported", flush=True)
    from src.decoder.metrics import compute_reconstruction_metrics, compute_error_map
    print("[run_decoder] metrics imported", flush=True)
    from src.decoder.vae_decoder import VAEDecoder
    print("[run_decoder] vae_decoder imported", flush=True)
    from src.decoder.vit_decoder import ViTDecoder
    print("[run_decoder] vit_decoder imported", flush=True)
    from src.decoder.vjepa_readout import VJEPAReadout
    print("[run_decoder] vjepa_readout imported", flush=True)
    from src.decoder.feature_extractor import (
        load_encoders,
        extract_features_from_image,
        save_latent_npz,
        load_latent_npz,
        extract_readout_training_data
    )
    print("[run_decoder] feature_extractor imported", flush=True)
except Exception as e:
    print("[run_decoder] Import failure:", flush=True)
    traceback.print_exc()
    sys.exit(1)


def run_pipeline(
    input_image_path: str,
    input_video_path: str,
    out_dir: str,
    force_recompute_features: bool = False
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cache_dir = PROJECT_ROOT / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_features_path = cache_dir / "extracted_latent_features.npz"

    print("=" * 75, flush=True)
    print("WORLD MODEL LATENT DECODER & GENERATIVE READOUT BENCHMARK", flush=True)
    print("=" * 75, flush=True)

    # 1. Load Ground Truth Image
    print(f"\n[Step 1] Loading ground truth input image from: {input_image_path}", flush=True)
    raw_img = Image.open(input_image_path).convert("RGB")
    gt_256 = np.array(raw_img.resize((256, 256), Image.Resampling.LANCZOS))

    # 2. Extract or Load Latent Features
    vae_enc, vit_enc, vjepa_enc = None, None, None
    if npz_features_path.exists() and not force_recompute_features:
        print(f"[Step 2] Found existing extracted features at: {npz_features_path}", flush=True)
        features = load_latent_npz(npz_features_path)
    else:
        print("[Step 2] Extracting latent features S_t from scratch...", flush=True)
        vae_enc, vit_enc, vjepa_enc = load_encoders(device="auto")
        features = extract_features_from_image(raw_img, vae_enc, vit_enc, vjepa_enc)
        save_latent_npz(features, npz_features_path)

    s_vae = features.get("s_vae") if "s_vae" in features else features["S_VAE"]
    s_vit = features.get("s_vit") if "s_vit" in features else features["S_ViT"]
    s_vjepa = features.get("s_vjepa") if "s_vjepa" in features else features["S_VJEPA"]

    print(f"Loaded feature shapes -> VAE: {s_vae.shape}, ViT: {s_vit.shape}, V-JEPA: {s_vjepa.shape}", flush=True)

    # 3. Decode S_VAE
    print("\n[Step 3] Running VAE continuous deconvolution decoder...", flush=True)
    vae_dec = VAEDecoder(device="auto")
    recon_vae = vae_dec.decode(s_vae, target_size=(256, 256))
    metrics_vae = compute_reconstruction_metrics(gt_256, recon_vae)
    print(f"  VAE Reconstruction -> PSNR: {metrics_vae['PSNR (dB)']:.2f} dB, SSIM: {metrics_vae['SSIM']:.4f}", flush=True)

    # 4. Prepare Readout Probes Data
    print("\n[Step 4] Extracting multi-frame readout probing data from video...", flush=True)
    if vit_enc is None or vjepa_enc is None:
        vae_enc, vit_enc, vjepa_enc = load_encoders(device="auto")

    vit_tokens_train, vit_patches_train, vjepa_tokens_train, vjepa_patches_train = extract_readout_training_data(
        video_path=input_video_path,
        vit_enc=vit_enc,
        vjepa_enc=vjepa_enc,
        max_frames=12
    )

    # 5. Fit & Decode ViT Readout
    print("\n[Step 5] Fitting MAE Linear Patch Readout for ViT...", flush=True)
    vit_dec = ViTDecoder(alpha=10.0)
    vit_dec.fit(vit_tokens_train, vit_patches_train)
    recon_vit = vit_dec.decode(s_vit, (256, 256))
    metrics_vit = compute_reconstruction_metrics(gt_256, recon_vit)
    print(f"  ViT Readout -> PSNR: {metrics_vit['PSNR (dB)']:.2f} dB, SSIM: {metrics_vit['SSIM']:.4f}", flush=True)

    # 6. Fit & Decode V-JEPA Readout
    print("\n[Step 6] Fitting Generative Readout Probe for Meta V-JEPA 2...", flush=True)
    vjepa_readout = VJEPAReadout(alpha=10.0)
    vjepa_readout.fit(vjepa_tokens_train, vjepa_patches_train)
    recon_vjepa = vjepa_readout.decode(s_vjepa, (256, 256))
    metrics_vjepa = compute_reconstruction_metrics(gt_256, recon_vjepa)
    print(f"  V-JEPA Readout -> PSNR: {metrics_vjepa['PSNR (dB)']:.2f} dB, SSIM: {metrics_vjepa['SSIM']:.4f}", flush=True)

    # 7. Save Metrics Table
    summary_df = pd.DataFrame([
        {
            "Model": "SD-VAE",
            "Latent Shape": str(s_vae.shape),
            "Decoder Architecture": "Native Continuous Deconvolution",
            "PSNR (dB)": metrics_vae["PSNR (dB)"],
            "SSIM": metrics_vae["SSIM"],
            "MSE": metrics_vae["MSE"],
            "MAE": metrics_vae["MAE"]
        },
        {
            "Model": "ViT-Base",
            "Latent Shape": str(s_vit.shape),
            "Decoder Architecture": "MAE-Style Linear Patch Probing",
            "PSNR (dB)": metrics_vit["PSNR (dB)"],
            "SSIM": metrics_vit["SSIM"],
            "MSE": metrics_vit["MSE"],
            "MAE": metrics_vit["MAE"]
        },
        {
            "Model": "Meta V-JEPA 2",
            "Latent Shape": str(s_vjepa.shape),
            "Decoder Architecture": "Generative Readout Probing",
            "PSNR (dB)": metrics_vjepa["PSNR (dB)"],
            "SSIM": metrics_vjepa["SSIM"],
            "MSE": metrics_vjepa["MSE"],
            "MAE": metrics_vjepa["MAE"]
        }
    ])
    csv_out_path = out_path / "05_reconstruction_metrics.csv"
    summary_df.to_csv(csv_out_path, index=False)
    print(f"\n[Step 7] Metrics saved to: {csv_out_path}", flush=True)
    print(summary_df.to_string(index=False), flush=True)

    # 8. Visualizations
    print("\n[Step 8] Generating visual artifacts...", flush=True)

    # 8.1 Individual Images
    Image.fromarray(recon_vae).save(out_path / "01_vae_reconstruction.png")
    Image.fromarray(recon_vit).save(out_path / "02_vit_reconstruction.png")
    Image.fromarray(recon_vjepa).save(out_path / "03_vjepa_readout.png")

    # 8.2 Side-by-side 4-panel comparison
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=180)
    panels = [
        ("Ground Truth (Canonical Frame 56)", gt_256, None),
        ("SD-VAE (Native Deconv)", recon_vae, metrics_vae),
        ("ViT-Base (MAE Linear Probe)", recon_vit, metrics_vit),
        ("Meta V-JEPA 2 (Generative Readout)", recon_vjepa, metrics_vjepa)
    ]
    for ax, (title, img, m) in zip(axes, panels):
        ax.imshow(img)
        if m is not None:
            ax.set_title(f"{title}\nPSNR: {m['PSNR (dB)']:.2f} dB | SSIM: {m['SSIM']:.4f}", fontsize=11, fontweight="bold")
        else:
            ax.set_title(f"{title}\nOriginal Reference", fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.suptitle("LATENT VISUAL RECONSTRUCTION BENCHMARK (GROUND TRUTH VS. DECODED LATENTS)", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig_4panel_path = out_path / "04_side_by_side_comparison.png"
    plt.savefig(fig_4panel_path, bbox_inches="tight", dpi=180)
    plt.close()
    print(f"  Saved 4-panel comparison to: {fig_4panel_path}", flush=True)

    # 8.3 Error Heatmaps
    err_vae = compute_error_map(gt_256, recon_vae)
    err_vit = compute_error_map(gt_256, recon_vit)
    err_vjepa = compute_error_map(gt_256, recon_vjepa)
    vmax = max(err_vae.max(), err_vit.max(), err_vjepa.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=180)
    err_panels = [
        ("SD-VAE Error Map", err_vae, metrics_vae),
        ("ViT-Base Error Map", err_vit, metrics_vit),
        ("Meta V-JEPA 2 Error Map", err_vjepa, metrics_vjepa)
    ]
    for ax, (title, err, m) in zip(axes, err_panels):
        im = ax.imshow(err, cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(f"{title}\nMSE: {m['MSE']:.1f} | MAE: {m['MAE']:.1f}", fontsize=11, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("ABSOLUTE RECONSTRUCTION ERROR HEATMAPS |I - \\hat{I}| (PIXEL OPTIMIZATION TRAP ANALYSIS)", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig_err_path = out_path / "05_reconstruction_error_heatmaps.png"
    plt.savefig(fig_err_path, bbox_inches="tight", dpi=180)
    plt.close()
    print(f"  Saved error heatmaps to: {fig_err_path}", flush=True)

    print("\n" + "=" * 75, flush=True)
    print(f"[COMPLETED] All reconstruction artifacts successfully written to: {out_path}", flush=True)
    print("=" * 75 + "\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="World Model Latent Reconstruction Pipeline")
    parser.add_argument("--image", type=str, default=str(PROJECT_ROOT / "data" / "raw" / "sample_image.jpg"), help="Path to input image")
    parser.add_argument("--video", type=str, default=str(PROJECT_ROOT / "data" / "raw" / "sample_video.mp4"), help="Path to input video")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "RESULT" / "04_decoder_pixel_optimization_trap"), help="Output directory")
    parser.add_argument("--force-recompute", action="store_true", help="Force recomputation of latent features")
    args = parser.parse_args()

    run_pipeline(
        input_image_path=args.image,
        input_video_path=args.video,
        out_dir=args.output,
        force_recompute_features=args.force_recompute
    )

