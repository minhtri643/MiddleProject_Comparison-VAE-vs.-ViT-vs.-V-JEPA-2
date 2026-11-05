"""
Feature extraction pipeline for VAE, ViT, and V-JEPA World Models.

Extracts continuous spatial latent state St and training patch pairs for Generative Readout.
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from typing import Dict, Tuple
import cv2
import numpy as np
from PIL import Image

# Middle_Project root is 3 levels up: Decoder/modules/feature_extractor.py -> Middle_Project
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import VAEEncoder, ViTEncoder, VJEPAEncoder
from .patch_utils import image_to_patches, resize_image


def load_encoders(device: str = "auto"):
    """Initialize all three encoders."""
    vae_enc = VAEEncoder(checkpoint="stabilityai/sd-vae-ft-mse", device=device)
    vit_enc = ViTEncoder(checkpoint="google/vit-base-patch16-224", device=device)
    vjepa_enc = VJEPAEncoder(checkpoint="facebook/vjepa2-vitl-fpc64-256", device=device, frames=4)
    return vae_enc, vit_enc, vjepa_enc


def extract_features_from_image(img: Image.Image, vae_enc, vit_enc, vjepa_enc) -> Dict[str, np.ndarray]:
    """
    Extract spatial latent state S_t for VAE, ViT, and V-JEPA from an input PIL Image.
    """
    print("[Extractor] Extracting S_VAE (32, 32, 4)...")
    s_vae = vae_enc.encode(img)
    
    print("[Extractor] Extracting S_ViT (14, 14, 768)...")
    s_vit = vit_enc.encode(img)
    
    print("[Extractor] Extracting S_VJEPA (16, 16, 1024)...")
    s_vjepa = vjepa_enc.encode(img)
    
    return {
        "S_VAE": s_vae,
        "S_ViT": s_vit,
        "S_VJEPA": s_vjepa
    }


def save_latent_npz(features: Dict[str, np.ndarray], out_path: str):
    """Save extracted features to compressed .npz archive."""
    np.savez_compressed(
        out_path,
        S_VAE=features["S_VAE"],
        S_ViT=features["S_ViT"],
        S_VJEPA=features["S_VJEPA"]
    )
    print(f"[Extractor] Saved latent features to {out_path}")


def load_latent_npz(in_path: str) -> Dict[str, np.ndarray]:
    """Load latent features from .npz archive."""
    data = np.load(in_path)
    return {k: data[k] for k in data.files}


def extract_readout_training_data(
    video_path: str,
    vit_enc,
    vjepa_enc,
    max_frames: int = 16
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample frames from the driving video and extract (token, patch) pairs
    to fit the MAE linear probe and V-JEPA generative readout.
    
    Returns:
        vit_tokens_all, vit_patches_all, vjepa_tokens_all, vjepa_patches_all
    """
    print(f"[Extractor] Loading frames from {video_path} (max_frames={max_frames})...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    
    frames = []
    current_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        current_idx += 1
    cap.release()
    
    print(f"[Extractor] Extracted {len(frames)} frames for probe calibration.")
    
    vit_tokens_list = []
    vit_patches_list = []
    vjepa_tokens_list = []
    vjepa_patches_list = []
    
    for i, frame in enumerate(frames):
        # 1. ViT data (224x224)
        frame_224 = resize_image(frame, (224, 224))
        vit_patches = image_to_patches(frame_224, patch_size=16)  # (196, 768)
        vit_grid = vit_enc.encode(frame_224)                     # (14, 14, 768)
        vit_tokens = vit_grid.reshape(14 * 14, 768)
        
        vit_tokens_list.append(vit_tokens)
        vit_patches_list.append(vit_patches)
        
        # 2. V-JEPA data (256x256)
        frame_256 = resize_image(frame, (256, 256))
        vjepa_patches = image_to_patches(frame_256, patch_size=16)  # (256, 768)
        vjepa_grid = vjepa_enc.encode(frame_256)                   # (16, 16, 1024)
        vjepa_tokens = vjepa_grid.reshape(16 * 16, 1024)
        
        vjepa_tokens_list.append(vjepa_tokens)
        vjepa_patches_list.append(vjepa_patches)
        
    vit_tokens_all = np.vstack(vit_tokens_list)
    vit_patches_all = np.vstack(vit_patches_list)
    vjepa_tokens_all = np.vstack(vjepa_tokens_list)
    vjepa_patches_all = np.vstack(vjepa_patches_list)
    
    print(f"[Extractor] ViT dataset: {vit_tokens_all.shape} -> {vit_patches_all.shape}")
    print(f"[Extractor] V-JEPA dataset: {vjepa_tokens_all.shape} -> {vjepa_patches_all.shape}")
    
    return vit_tokens_all, vit_patches_all, vjepa_tokens_all, vjepa_patches_all
