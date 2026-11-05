"""
Image Reconstruction Quality Metrics and Error Heatmaps.

  Image quality assessment: from error visibility to structural similarity.
  IEEE Transactions on Image Processing, 13(4), 600-612.
"""

from typing import Dict, Tuple
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn


def compute_reconstruction_metrics(original_rgb: np.ndarray, reconstructed_rgb: np.ndarray) -> Dict[str, float]:
    """
    Compute rigorous reconstruction metrics between target and decoded RGB images.
    
    Args:
        original_rgb: ground truth uint8 array of shape (H, W, 3)
        reconstructed_rgb: decoded uint8 array of shape (H, W, 3)
        
    Returns:
        metrics_dict containing PSNR (dB), SSIM, MSE, and MAE.
    """
    if original_rgb.shape != reconstructed_rgb.shape:
        raise ValueError(f"Shape mismatch: original {original_rgb.shape} vs reconstructed {reconstructed_rgb.shape}")

    orig_float = original_rgb.astype(np.float64)
    recon_float = reconstructed_rgb.astype(np.float64)
    
    # Mean Squared Error (MSE)
    mse = float(np.mean((orig_float - recon_float) ** 2))
    
    # Mean Absolute Error (MAE / L1)
    mae = float(np.mean(np.abs(orig_float - recon_float)))
    
    # Peak Signal-to-Noise Ratio (PSNR in dB)
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = float(psnr_fn(original_rgb, reconstructed_rgb, data_range=255))
        
    # Structural Similarity Index (SSIM)
    ssim_val = float(ssim_fn(original_rgb, reconstructed_rgb, channel_axis=2, data_range=255))
    
    return {
        "PSNR (dB)": round(psnr, 2),
        "SSIM": round(ssim_val, 4),
        "MSE": round(mse, 2),
        "MAE": round(mae, 2),
        "MAE (L1)": round(mae, 2)
    }


def compute_error_map(original_rgb: np.ndarray, reconstructed_rgb: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel absolute error map averaged over color channels.
    
    Returns:
        error_map: 2D float array of shape (H, W) in range [0, 255].
    """
    orig_float = original_rgb.astype(np.float32)
    recon_float = reconstructed_rgb.astype(np.float32)
    error_map = np.mean(np.abs(orig_float - recon_float), axis=-1)
    return error_map
