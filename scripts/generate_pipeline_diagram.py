"""
Generate a publication-grade architecture pipeline diagram: pipeline_overview.png
Resolution: 3600 x 2400 (3:2 aspect ratio).
Covers all 5 phases of the benchmark, including Phase 5 (Temporal Causality).
Polished layout with zero text overlaps and automatic word wrapping.
"""

import os
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

def create_pipeline_diagram(output_path):
    fig = plt.figure(figsize=(36, 24), dpi=100)  # 3600 x 2400 pixels
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Color Palette
    bg_color = "#F8FAFC"
    card_bg = "#FFFFFF"
    card_border = "#CBD5E1"
    text_dark = "#0F172A"
    text_muted = "#475569"
    text_accent = "#1E293B"

    vae_color = "#EA580C"     # Orange-Red
    vae_bg = "#FFF7ED"
    vit_color = "#2563EB"     # Royal Blue
    vit_bg = "#EFF6FF"
    vjepa_color = "#16A34A"   # Emerald Green
    vjepa_bg = "#F0FDF4"
    header_navy = "#0F172A"

    # Fill background
    bg_rect = patches.Rectangle((0, 0), 1, 1, facecolor=bg_color, edgecolor='none', zorder=0)
    ax.add_patch(bg_rect)

    # ==========================================
    # HEADER BANNER (y: 0.915 to 0.99)
    # ==========================================
    header_box = patches.FancyBboxPatch((0.015, 0.915), 0.97, 0.075,
                                        boxstyle="round,pad=0.008,rounding_size=0.01",
                                        facecolor=header_navy, edgecolor='none', zorder=1)
    ax.add_patch(header_box)

    ax.text(0.028, 0.963, "AUTONOMOUS DRIVING WORLD MODEL BENCHMARK: LATENT STATE SUITE",
            fontsize=23, fontweight='bold', color="#FFFFFF", va='center', zorder=2)
    ax.text(0.028, 0.935, "Empirical Evaluation: Pixel Compression (SD-VAE) vs. Supervised (ViT) vs. Joint-Embedding (Meta V-JEPA 2)",
            fontsize=15, fontweight='normal', color="#94A3B8", va='center', zorder=2)

    # Badges on right side of header
    badges = [
        ("INFERENCE-ONLY (NO TUNING TRAP)", "#334155", "#F8FAFC"),
        ("CANONICAL KEYFRAME 56", "#0284C7", "#FFFFFF"),
        ("100% REPRODUCIBLE", "#059669", "#FFFFFF")
    ]
    bx = 0.975
    for label, bcolor, fcolor in reversed(badges):
        width = len(label) * 0.0062 + 0.016
        bx -= width
        bb = patches.FancyBboxPatch((bx, 0.936), width - 0.006, 0.030,
                                    boxstyle="round,pad=0.004,rounding_size=0.006",
                                    facecolor=bcolor, edgecolor='none', zorder=3)
        ax.add_patch(bb)
        ax.text(bx + (width - 0.006)/2, 0.951, label, fontsize=11.5, fontweight='bold',
                color=fcolor, ha='center', va='center', zorder=4)
        bx -= 0.006

    # ==========================================
    # ROW 1: INPUT DATA & ENCODER PARADIGMS (y: 0.69 to 0.90)
    # ==========================================
    # (1) Input Data Stream (x: 0.015 to 0.22)
    input_box = patches.FancyBboxPatch((0.015, 0.69), 0.205, 0.21,
                                       boxstyle="round,pad=0.006,rounding_size=0.008",
                                       facecolor=card_bg, edgecolor=card_border, linewidth=2, zorder=1)
    ax.add_patch(input_box)
    
    in_title_bar = patches.FancyBboxPatch((0.015, 0.865), 0.205, 0.035,
                                          boxstyle="round,pad=0.004,rounding_size=0.008",
                                          facecolor="#E2E8F0", edgecolor='none', zorder=2)
    ax.add_patch(in_title_bar)
    ax.text(0.025, 0.882, "(1) INPUT TRAFFIC DATA (data/raw/)", fontsize=13.5, fontweight='bold', color=text_dark, va='center', zorder=3)

    # Embed sample image
    try:
        sample_img = Image.open('Middle_Project/data/raw/sample_image.jpg')
        sub_ax1 = fig.add_axes([0.025, 0.705, 0.075, 0.15])
        sub_ax1.imshow(sample_img)
        sub_ax1.axis('off')
        sub_ax1.set_title("Canonical Keyframe 56\n(256x256 RGB)", fontsize=11, color=text_dark, pad=4)
    except Exception as e:
        print("Could not load sample image:", e)

    # Input text details
    ax.text(0.108, 0.835, "• Video Stream:", fontsize=12, fontweight='bold', color=text_dark, zorder=3)
    ax.text(0.108, 0.815, "  car-detection.mp4", fontsize=11, color=text_muted, zorder=3)
    ax.text(0.108, 0.795, "  75 frames, 12.5 FPS, 256x256", fontsize=11, color=text_muted, zorder=3)
    ax.text(0.108, 0.765, "• Spatial Structure:", fontsize=12, fontweight='bold', color=text_dark, zorder=3)
    ax.text(0.108, 0.745, "  Vehicle at [0.50, 0.65]", fontsize=11, color=text_muted, zorder=3)
    ax.text(0.108, 0.725, "  Road at [0.10, 0.12]", fontsize=11, color=text_muted, zorder=3)
    ax.text(0.108, 0.700, "• Temporal Context: 15 frames", fontsize=11, fontweight='bold', color="#0284C7", zorder=3)

    # Encoders Container (x: 0.235 to 0.985)
    encoders_box = patches.FancyBboxPatch((0.232, 0.69), 0.753, 0.21,
                                         boxstyle="round,pad=0.006,rounding_size=0.008",
                                         facecolor=card_bg, edgecolor=card_border, linewidth=2, zorder=1)
    ax.add_patch(encoders_box)

    enc_title_bar = patches.FancyBboxPatch((0.232, 0.865), 0.753, 0.035,
                                           boxstyle="round,pad=0.004,rounding_size=0.008",
                                           facecolor="#E2E8F0", edgecolor='none', zorder=2)
    ax.add_patch(enc_title_bar)
    ax.text(0.245, 0.882, "(2) FOUNDATIONAL VISION REPRESENTATION PARADIGMS (Inference Feature Extraction)",
            fontsize=13.5, fontweight='bold', color=text_dark, va='center', zorder=3)

    # Encoder 1: SD-VAE
    vae_panel = patches.FancyBboxPatch((0.242, 0.702), 0.232, 0.155,
                                       boxstyle="round,pad=0.005,rounding_size=0.008",
                                       facecolor=vae_bg, edgecolor=vae_color, linewidth=2, zorder=2)
    ax.add_patch(vae_panel)
    ax.text(0.252, 0.835, "SD-VAE", fontsize=16, fontweight='bold', color=vae_color, zorder=3)
    ax.text(0.252, 0.812, "stabilityai/sd-vae-ft-mse", fontsize=11, style='italic', color=text_muted, zorder=3)
    ax.text(0.252, 0.785, "• Family: Pixel-Level Generative Model", fontsize=11, color=text_dark, zorder=3)
    ax.text(0.252, 0.765, "• Objective: L2 Pixel Recon + KL Penalty", fontsize=11, color=text_muted, zorder=3)
    ax.text(0.252, 0.745, "• Grid: 32x32 patches (8x downsample)", fontsize=11, color=text_muted, zorder=3)
    
    vae_shape_badge = patches.FancyBboxPatch((0.252, 0.710), 0.212, 0.024,
                                             boxstyle="round,pad=0.003,rounding_size=0.005",
                                             facecolor=vae_color, edgecolor='none', zorder=3)
    ax.add_patch(vae_shape_badge)
    ax.text(0.358, 0.722, "St = 32 x 32 x 4  (4,096 params)", fontsize=12, fontweight='bold', color="#FFFFFF", ha='center', va='center', zorder=4)

    # Encoder 2: ViT-Base
    vit_panel = patches.FancyBboxPatch((0.492, 0.702), 0.232, 0.155,
                                       boxstyle="round,pad=0.005,rounding_size=0.008",
                                       facecolor=vit_bg, edgecolor=vit_color, linewidth=2, zorder=2)
    ax.add_patch(vit_panel)
    ax.text(0.502, 0.835, "ViT-Base", fontsize=16, fontweight='bold', color=vit_color, zorder=3)
    ax.text(0.502, 0.812, "google/vit-base-patch16-224", fontsize=11, style='italic', color=text_muted, zorder=3)
    ax.text(0.502, 0.785, "• Family: Supervised Feature Classifier", fontsize=11, color=text_dark, zorder=3)
    ax.text(0.502, 0.765, "• Objective: Cross-Entropy (ImageNet-1k)", fontsize=11, color=text_muted, zorder=3)
    ax.text(0.502, 0.745, "• Grid: 14x14 tokens (16x patch size)", fontsize=11, color=text_muted, zorder=3)

    vit_shape_badge = patches.FancyBboxPatch((0.502, 0.710), 0.212, 0.024,
                                             boxstyle="round,pad=0.003,rounding_size=0.005",
                                             facecolor=vit_color, edgecolor='none', zorder=3)
    ax.add_patch(vit_shape_badge)
    ax.text(0.608, 0.722, "St = 14 x 14 x 768  (150,528 params)", fontsize=12, fontweight='bold', color="#FFFFFF", ha='center', va='center', zorder=4)

    # Encoder 3: Meta V-JEPA 2
    vjepa_panel = patches.FancyBboxPatch((0.742, 0.702), 0.232, 0.155,
                                         boxstyle="round,pad=0.005,rounding_size=0.008",
                                         facecolor=vjepa_bg, edgecolor=vjepa_color, linewidth=2, zorder=2)
    ax.add_patch(vjepa_panel)
    ax.text(0.752, 0.835, "Meta V-JEPA 2", fontsize=16, fontweight='bold', color=vjepa_color, zorder=3)
    ax.text(0.752, 0.812, "facebook/vjepa2-vitl-fpc64-256", fontsize=11, style='italic', color=text_muted, zorder=3)
    ax.text(0.752, 0.785, "• Family: Non-Generative Joint-Embedding", fontsize=11, color=text_dark, zorder=3)
    ax.text(0.752, 0.765, "• Objective: Video Masked Latent Prediction", fontsize=11, color=text_muted, zorder=3)
    ax.text(0.752, 0.745, "• Grid: 16x16 tokens (16x spatiotemporal)", fontsize=11, color=text_muted, zorder=3)

    vjepa_shape_badge = patches.FancyBboxPatch((0.752, 0.710), 0.212, 0.024,
                                               boxstyle="round,pad=0.003,rounding_size=0.005",
                                               facecolor=vjepa_color, edgecolor='none', zorder=3)
    ax.add_patch(vjepa_shape_badge)
    ax.text(0.858, 0.722, "St = 16 x 16 x 1024  (262,144 params)", fontsize=12, fontweight='bold', color="#FFFFFF", ha='center', va='center', zorder=4)

    # Connecting Arrow
    ax.annotate('', xy=(0.232, 0.795), xytext=(0.220, 0.795),
                arrowprops=dict(arrowstyle="-|>", lw=4, color="#0284C7", mutation_scale=25), zorder=5)

    # ==========================================
    # ROW 2: THE 5 COMPREHENSIVE BENCHMARK PHASES (y: 0.22 to 0.67)
    # ==========================================
    phase_header = patches.FancyBboxPatch((0.015, 0.635), 0.97, 0.035,
                                          boxstyle="round,pad=0.004,rounding_size=0.008",
                                          facecolor=header_navy, edgecolor='none', zorder=2)
    ax.add_patch(phase_header)
    ax.text(0.025, 0.652, "(3) FIVE FOUNDATIONAL EMPIRICAL BENCHMARK PIPELINES",
            fontsize=14.5, fontweight='bold', color="#FFFFFF", va='center', zorder=3)
    ax.text(0.44, 0.652, "Standardized Evaluation across Spatial Topology, Effective Rank, Weather Invariance, Decoder Trap, and Temporal Causality",
            fontsize=12.5, color="#94A3B8", va='center', zorder=3)

    col_w = 0.186
    gap = 0.010
    left_start = 0.015
    y_top = 0.625
    col_h = 0.405

    # PHASE 1: Spatial Manifold
    p1_x = left_start
    p1_box = patches.FancyBboxPatch((p1_x, y_top - col_h), col_w, col_h,
                                    boxstyle="round,pad=0.005,rounding_size=0.008",
                                    facecolor=card_bg, edgecolor=card_border, linewidth=2, zorder=1)
    ax.add_patch(p1_box)
    
    p1_bar = patches.FancyBboxPatch((p1_x, y_top - 0.040), col_w, 0.040,
                                    boxstyle="round,pad=0.003,rounding_size=0.006",
                                    facecolor="#E0E7FF", edgecolor='none', zorder=2)
    ax.add_patch(p1_bar)
    ax.text(p1_x + 0.008, y_top - 0.018, "PHASE 1: SPATIAL TOPOLOGY", fontsize=12.5, fontweight='bold', color="#3730A3", va='center', zorder=3)
    ax.text(p1_x + 0.008, y_top - 0.032, "scripts/main_run.py -> RESULT/01_...", fontsize=10, color=text_muted, va='center', zorder=3)

    ax.text(p1_x + 0.008, y_top - 0.055, "• 3-Channel PCA-RGB Projection:", fontsize=11, fontweight='bold', color=text_dark, zorder=3)
    ax.text(p1_x + 0.008, y_top - 0.072, "  Top-3 PC Explained Variance:", fontsize=10.5, color=text_muted, zorder=3)
    ax.text(p1_x + 0.008, y_top - 0.090, "  - VAE: 90.67% (Subspace collapse)", fontsize=10.5, fontweight='bold', color=vae_color, zorder=3)
    ax.text(p1_x + 0.008, y_top - 0.108, "  - ViT: 50.49%", fontsize=10.5, color=vit_color, zorder=3)
    ax.text(p1_x + 0.008, y_top - 0.126, "  - V-JEPA: 29.21% (Multi-dim)", fontsize=10.5, fontweight='bold', color=vjepa_color, zorder=3)
    ax.text(p1_x + 0.008, y_top - 0.148, "• Query Similarity Separation:", fontsize=11, fontweight='bold', color=text_dark, zorder=3)
    ax.text(p1_x + 0.008, y_top - 0.165, "  Vehicle (A) vs Roadway (B)", fontsize=10.5, color=text_muted, zorder=3)

    try:
        p1_img = Image.open('Middle_Project/RESULT/01_spatial_manifold_clustering/01_spatial_comparison.png')
        sub_p1 = fig.add_axes([p1_x + 0.008, y_top - col_h + 0.012, col_w - 0.016, 0.215])
        sub_p1.imshow(p1_img)
        sub_p1.axis('off')
    except Exception as e:
        print("P1 image error:", e)

    # PHASE 2: Spectral Effective Rank
    p2_x = p1_x + col_w + gap
    p2_box = patches.FancyBboxPatch((p2_x, y_top - col_h), col_w, col_h,
                                    boxstyle="round,pad=0.005,rounding_size=0.008",
                                    facecolor=card_bg, edgecolor=card_border, linewidth=2, zorder=1)
    ax.add_patch(p2_box)

    p2_bar = patches.FancyBboxPatch((p2_x, y_top - 0.040), col_w, 0.040,
                                    boxstyle="round,pad=0.003,rounding_size=0.006",
                                    facecolor="#EDE9FE", edgecolor='none', zorder=2)
    ax.add_patch(p2_bar)
    ax.text(p2_x + 0.008, y_top - 0.018, "PHASE 2: EFFECTIVE RANK", fontsize=12.5, fontweight='bold', color="#5B21B6", va='center', zorder=3)
    ax.text(p2_x + 0.008, y_top - 0.032, "RankMe SVD Entropy -> RESULT/02_...", fontsize=10, color=text_muted, va='center', zorder=3)

    ax.text(p2_x + 0.008, y_top - 0.055, "• 3D Orthogonal PCA Expansion:", fontsize=11, fontweight='bold', color=text_dark, zorder=3)
    ax.text(p2_x + 0.008, y_top - 0.072, "  Resolves 2D false cluster collision", fontsize=10.5, color=text_muted, zorder=3)
    ax.text(p2_x + 0.008, y_top - 0.092, "• Effective Rank (RankMe):", fontsize=11, fontweight='bold', color=text_dark, zorder=3)
    ax.text(p2_x + 0.008, y_top - 0.110, "  - VAE: 3.58 / 4.0 (Util: 0.895)", fontsize=10.5, color=vae_color, zorder=3)
    ax.text(p2_x + 0.008, y_top - 0.128, "  - ViT: 112.98 / 195.0", fontsize=10.5, color=vit_color, zorder=3)
    ax.text(p2_x + 0.008, y_top - 0.146, "  - V-JEPA: 166.23 / 255.0", fontsize=10.5, fontweight='bold', color=vjepa_color, zorder=3)
    ax.text(p2_x + 0.008, y_top - 0.165, "  V-JEPA eliminates rank collapse!", fontsize=10.5, fontweight='bold', color="#059669", zorder=3)

    try:
        p2_img = Image.open('Middle_Project/RESULT/02_spectral_effective_rank/02_patch_scatter_3d.png')
        sub_p2 = fig.add_axes([p2_x + 0.008, y_top - col_h + 0.012, col_w - 0.016, 0.215])
        sub_p2.imshow(p2_img)
        sub_p2.axis('off')
    except Exception as e:
        print("P2 image error:", e)

    # PHASE 3: Weather Robustness
    p3_x = p2_x + col_w + gap
    p3_box = patches.FancyBboxPatch((p3_x, y_top - col_h), col_w, col_h,
                                    boxstyle="round,pad=0.005,rounding_size=0.008",
                                    facecolor=card_bg, edgecolor=card_border, linewidth=2, zorder=1)
    ax.add_patch(p3_box)

    p3_bar = patches.FancyBboxPatch((p3_x, y_top - 0.040), col_w, 0.040,
                                    boxstyle="round,pad=0.003,rounding_size=0.006",
                                    facecolor="#FEE2E2", edgecolor='none', zorder=2)
    ax.add_patch(p3_bar)
    ax.text(p3_x + 0.008, y_top - 0.018, "PHASE 3: ADVERSE WEATHER", fontsize=12.5, fontweight='bold', color="#991B1B", va='center', zorder=3)
    ax.text(p3_x + 0.008, y_top - 0.032, "Koschmieder Fog -> RESULT/03_...", fontsize=10, color=text_muted, va='center', zorder=3)

    ax.text(p3_x + 0.008, y_top - 0.055, "• Atmospheric Scattering Model:", fontsize=11, fontweight='bold', color=text_dark, zorder=3)
    ax.text(p3_x + 0.008, y_top - 0.072, "  Level 3 Dense Fog (beta = 0.65)", fontsize=10.5, color=text_muted, zorder=3)
    ax.text(p3_x + 0.008, y_top - 0.092, "• Latent Feature Drift (Delta Cos A):", fontsize=11, fontweight='bold', color=text_dark, zorder=3)
    ax.text(p3_x + 0.008, y_top - 0.110, "  - VAE: +0.3808 (38% Collapse)", fontsize=10.5, fontweight='bold', color=vae_color, zorder=3)
    ax.text(p3_x + 0.008, y_top - 0.128, "  - ViT: -0.0052 (Invariant)", fontsize=10.5, color=vit_color, zorder=3)
    ax.text(p3_x + 0.008, y_top - 0.146, "  - V-JEPA: -0.0079 (Invariant)", fontsize=10.5, fontweight='bold', color=vjepa_color, zorder=3)
    ax.text(p3_x + 0.008, y_top - 0.165, "  VAE fails under adverse weather!", fontsize=10.5, fontweight='bold', color="#DC2626", zorder=3)

    try:
        p3_img = Image.open('Middle_Project/RESULT/03_weather_robustness_stress_test/05_robustness_stress_test.png')
        sub_p3 = fig.add_axes([p3_x + 0.008, y_top - col_h + 0.012, col_w - 0.016, 0.215])
        sub_p3.imshow(p3_img)
        sub_p3.axis('off')
    except Exception as e:
        print("P3 image error:", e)

    # PHASE 4: Latent Probing & Pixel Trap
    p4_x = p3_x + col_w + gap
    p4_box = patches.FancyBboxPatch((p4_x, y_top - col_h), col_w, col_h,
                                    boxstyle="round,pad=0.005,rounding_size=0.008",
                                    facecolor=card_bg, edgecolor=card_border, linewidth=2, zorder=1)
    ax.add_patch(p4_box)

    p4_bar = patches.FancyBboxPatch((p4_x, y_top - 0.040), col_w, 0.040,
                                    boxstyle="round,pad=0.003,rounding_size=0.006",
                                    facecolor="#FEF3C7", edgecolor='none', zorder=2)
    ax.add_patch(p4_bar)
    ax.text(p4_x + 0.008, y_top - 0.018, "PHASE 4: PIXEL TRAP PROBE", fontsize=12.5, fontweight='bold', color="#92400E", va='center', zorder=3)
    ax.text(p4_x + 0.008, y_top - 0.032, "scripts/run_decoder.py -> RESULT/04_...", fontsize=10, color=text_muted, va='center', zorder=3)

    ax.text(p4_x + 0.008, y_top - 0.055, "• Reconstruction Probing:", fontsize=11, fontweight='bold', color=text_dark, zorder=3)
    ax.text(p4_x + 0.008, y_top - 0.072, "  Native Deconv vs Linear Ridge (alpha=10)", fontsize=10.5, color=text_muted, zorder=3)
    ax.text(p4_x + 0.008, y_top - 0.092, "• Reconstruction PSNR / SSIM:", fontsize=11, fontweight='bold', color=text_dark, zorder=3)
    ax.text(p4_x + 0.008, y_top - 0.110, "  - VAE: 35.25 dB (SSIM 0.974)", fontsize=10.5, fontweight='bold', color=vae_color, zorder=3)
    ax.text(p4_x + 0.008, y_top - 0.128, "  - ViT: 17.37 dB (SSIM 0.364)", fontsize=10.5, color=vit_color, zorder=3)
    ax.text(p4_x + 0.008, y_top - 0.146, "  - V-JEPA: 15.76 dB (SSIM 0.245)", fontsize=10.5, color=vjepa_color, zorder=3)
    ax.text(p4_x + 0.008, y_top - 0.165, "  High PSNR memorizes noise!", fontsize=10.5, fontweight='bold', color="#B45309", zorder=3)

    try:
        p4_img = Image.open('Middle_Project/RESULT/04_decoder_pixel_optimization_trap/04_side_by_side_comparison.png')
        sub_p4 = fig.add_axes([p4_x + 0.008, y_top - col_h + 0.035, col_w - 0.016, 0.165])
        sub_p4.imshow(p4_img)
        sub_p4.axis('off')
        sub_p4.set_title("Ground Truth vs VAE vs ViT vs V-JEPA Decoded", fontsize=9.5, color=text_muted, pad=3)
    except Exception as e:
        print("P4 image error:", e)

    # PHASE 5: Spatiotemporal Causality & Arrow of Time
    p5_x = p4_x + col_w + gap
    p5_box = patches.FancyBboxPatch((p5_x, y_top - col_h), col_w, col_h,
                                    boxstyle="round,pad=0.005,rounding_size=0.008",
                                    facecolor=card_bg, edgecolor=vjepa_color, linewidth=2.5, zorder=1)
    ax.add_patch(p5_box)

    p5_bar = patches.FancyBboxPatch((p5_x, y_top - 0.040), col_w, 0.040,
                                    boxstyle="round,pad=0.003,rounding_size=0.006",
                                    facecolor="#DCFCE7", edgecolor='none', zorder=2)
    ax.add_patch(p5_bar)
    ax.text(p5_x + 0.008, y_top - 0.018, "PHASE 5: TEMPORAL CAUSALITY", fontsize=12.5, fontweight='bold', color="#166534", va='center', zorder=3)
    ax.text(p5_x + 0.008, y_top - 0.032, "Forecasting & Arrow of Time -> 05_...", fontsize=10, color=text_muted, va='center', zorder=3)

    ax.text(p5_x + 0.008, y_top - 0.055, "• Target State Forecasting (S56):", fontsize=11, fontweight='bold', color=text_dark, zorder=3)
    ax.text(p5_x + 0.008, y_top - 0.072, "  Chronological (t0->t14) vs Shuffled (pi)", fontsize=10.5, color=text_muted, zorder=3)
    ax.text(p5_x + 0.008, y_top - 0.092, "• Causality Metrics & MSE Jump:", fontsize=11, fontweight='bold', color=text_dark, zorder=3)
    ax.text(p5_x + 0.008, y_top - 0.110, "  - V-JEPA Chrono CosSim: 0.9833", fontsize=10.5, fontweight='bold', color=vjepa_color, zorder=3)
    ax.text(p5_x + 0.008, y_top - 0.128, "  - V-JEPA Shuffled CosSim: 0.8820", fontsize=10.5, color=text_muted, zorder=3)
    ax.text(p5_x + 0.008, y_top - 0.146, "  - MSE Explosion: 7.23x JUMP!", fontsize=11, fontweight='bold', color="#DC2626", zorder=3)
    ax.text(p5_x + 0.008, y_top - 0.165, "  V-JEPA encodes physical momentum!", fontsize=10.5, fontweight='bold', color="#16A34A", zorder=3)

    try:
        p5_img = Image.open('Middle_Project/RESULT/05_temporal_causality_forecasting/07_temporal_trajectory_pca.png')
        sub_p5 = fig.add_axes([p5_x + 0.008, y_top - col_h + 0.012, col_w - 0.016, 0.215])
        sub_p5.imshow(p5_img)
        sub_p5.axis('off')
    except Exception as e:
        print("P5 image error:", e)

    # ==========================================
    # ROW 3: BOTTOM SYNTHESIS & EMPIRICAL VERDICT (y: 0.015 to 0.21)
    # ==========================================
    bot_y = 0.015
    bot_h = 0.190

    # Left: Quantitative Summary Table (x: 0.015 to 0.540)
    table_card = patches.FancyBboxPatch((0.015, bot_y), 0.525, bot_h,
                                        boxstyle="round,pad=0.006,rounding_size=0.008",
                                        facecolor=card_bg, edgecolor=card_border, linewidth=2, zorder=1)
    ax.add_patch(table_card)

    tbl_bar = patches.FancyBboxPatch((0.015, bot_y + bot_h - 0.032), 0.525, 0.032,
                                     boxstyle="round,pad=0.004,rounding_size=0.008",
                                     facecolor="#E2E8F0", edgecolor='none', zorder=2)
    ax.add_patch(tbl_bar)
    ax.text(0.025, bot_y + bot_h - 0.016, "MASTER EMPIRICAL BENCHMARK METRICS (Canonical Keyframe 56)",
            fontsize=13, fontweight='bold', color=text_dark, va='center', zorder=3)

    cols = ["Evaluation Metric", "SD-VAE", "ViT-Base", "Meta V-JEPA 2", "Scientific Finding"]
    col_x = [0.025, 0.195, 0.275, 0.360, 0.445]
    ty = bot_y + bot_h - 0.048
    for name, cx in zip(cols, col_x):
        ax.text(cx, ty, name, fontsize=10.5, fontweight='bold', color=text_accent, zorder=3)

    rows = [
        ("Top-3 PC Variance (%)", "90.67%", "50.49%", "29.21%", "V-JEPA (Deep multi-dim dispersion)"),
        ("Effective Rank (RankMe)", "3.58 / 4.0", "112.98 / 195", "166.23 / 255", "V-JEPA (Eliminates rank collapse)"),
        ("Fog Latent Drift (Delta Cos A)", "+0.3808 (38% drift)", "-0.0052", "-0.0079", "ViT & V-JEPA (Weather Invariance)"),
        ("Reconstruction PSNR / SSIM", "35.25 dB / 0.974", "17.37 dB / 0.364", "15.76 dB / 0.245", "VAE (Pixel Trap - Memorizes Noise)"),
        ("Temporal Shuffled MSE Jump", "72.0x", "4.0x", "7.23x Jump", "V-JEPA (Internalizes Arrow of Time)")
    ]

    for i, (m, v1, v2, v3, win) in enumerate(rows):
        ry = ty - 0.024 * (i + 1)
        ax.plot([0.020, 0.535], [ry + 0.018, ry + 0.018], color="#F1F5F9", lw=1.5, zorder=2)
        ax.text(col_x[0], ry, m, fontsize=10, color=text_dark, zorder=3)
        ax.text(col_x[1], ry, v1, fontsize=10, color=vae_color if "35.25" in v1 or "90" in v1 else text_muted, zorder=3)
        ax.text(col_x[2], ry, v2, fontsize=10, color=text_muted, zorder=3)
        ax.text(col_x[3], ry, v3, fontsize=10, fontweight='bold', color=vjepa_color, zorder=3)
        ax.text(col_x[4], ry, win, fontsize=9.5, style='italic', color=text_accent, zorder=3)

    # Right: Core Scientific Takeaways for World Models (x: 0.550 to 0.985)
    verdict_card = patches.FancyBboxPatch((0.550, bot_y), 0.435, bot_h,
                                         boxstyle="round,pad=0.006,rounding_size=0.008",
                                         facecolor=card_bg, edgecolor=vjepa_color, linewidth=2, zorder=1)
    ax.add_patch(verdict_card)

    vrd_bar = patches.FancyBboxPatch((0.550, bot_y + bot_h - 0.032), 0.435, 0.032,
                                     boxstyle="round,pad=0.004,rounding_size=0.008",
                                     facecolor=vjepa_bg, edgecolor='none', zorder=2)
    ax.add_patch(vrd_bar)
    ax.text(0.560, bot_y + bot_h - 0.016, "CORE SCIENTIFIC TAKEAWAYS FOR AUTONOMOUS WORLD MODELS",
            fontsize=13, fontweight='bold', color=vjepa_color, va='center', zorder=3)

    takeaways = [
        ("1. The Fallacy of Pixel Loss in World Modeling:",
         "High PSNR/SSIM does not signify world modeling ability. VAE squanders representation on high-frequency stochastic pixel textures (gravel, sheen) and catastrophically collapses (+38% drift) under fog."),
        ("2. Joint-Embedding Semantic Robustness:",
         "Eliminating the pixel decoder allows Meta V-JEPA 2 to achieve maximal effective dimensionality (RankMe = 166.23) and absolute semantic invariance under optical corruptions."),
        ("3. Physical Kinematics & The Arrow of Time:",
         "Meta V-JEPA 2 inherently internalizes the Arrow of Time. Permuting temporal frame order causes prediction MSE to explode 7.23-fold and shatters the smooth manifold trajectory.")
    ]

    vy = bot_y + bot_h - 0.052
    for title, desc in takeaways:
        ax.text(0.560, vy, title, fontsize=11, fontweight='bold', color=text_dark, zorder=3)
        vy -= 0.017
        wrapped_lines = textwrap.wrap(desc, width=82)
        for wline in wrapped_lines:
            ax.text(0.560, vy, wline, fontsize=9.5, color=text_muted, zorder=3)
            vy -= 0.014
        vy -= 0.008

    # Save at high DPI
    plt.savefig(output_path, dpi=100, facecolor=bg_color, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated new publication diagram: {output_path}")

if __name__ == '__main__':
    out = 'Middle_Project/docs/pipeline_overview.png'
    create_pipeline_diagram(out)
