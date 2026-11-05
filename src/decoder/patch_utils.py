"""
Patch extraction and image reassembly utilities.

"""

import numpy as np
from PIL import Image


def image_to_patches(img_arr: np.ndarray, patch_size: int = 16) -> np.ndarray:
    """
    Extract non-overlapping patches from an image array.
    
    Args:
        img_arr: np.ndarray of shape (H, W, 3)
        patch_size: spatial patch dimension (default 16)
        
    Returns:
        patches: np.ndarray of shape (N_patches, patch_size * patch_size * 3)
                 where N_patches = (H // patch_size) * (W // patch_size)
    """
    H, W, C = img_arr.shape
    assert H % patch_size == 0 and W % patch_size == 0, f"Image dimensions ({H}, {W}) must be divisible by {patch_size}"
    
    ny = H // patch_size
    nx = W // patch_size
    
    # Reshape to (ny, patch_size, nx, patch_size, C)
    patches = img_arr.reshape(ny, patch_size, nx, patch_size, C)
    # Transpose to (ny, nx, patch_size, patch_size, C)
    patches = patches.transpose(0, 2, 1, 3, 4)
    # Flatten patches to (ny * nx, patch_size * patch_size * C)
    patches_flat = patches.reshape(ny * nx, patch_size * patch_size * C)
    return patches_flat


def patches_to_image(patches_flat: np.ndarray, target_h: int, target_w: int, patch_size: int = 16) -> np.ndarray:
    """
    Reassemble flattened patches into a full (H, W, 3) image array.
    
    Args:
        patches_flat: np.ndarray of shape (N_patches, patch_size * patch_size * 3)
        target_h: reconstructed image height
        target_w: reconstructed image width
        patch_size: spatial patch dimension (default 16)
        
    Returns:
        img_arr: np.ndarray of shape (target_h, target_w, 3) uint8
    """
    ny = target_h // patch_size
    nx = target_w // patch_size
    C = 3
    
    expected_patches = ny * nx
    if patches_flat.shape[0] != expected_patches:
        raise ValueError(f"Patch count mismatch: expected {expected_patches}, got {patches_flat.shape[0]}")
        
    # Reshape to (ny, nx, patch_size, patch_size, C)
    patches = patches_flat.reshape(ny, nx, patch_size, patch_size, C)
    # Transpose to (ny, patch_size, nx, patch_size, C)
    img_arr = patches.transpose(0, 2, 1, 3, 4).reshape(target_h, target_w, C)
    # Clip and convert to uint8
    return np.clip(img_arr, 0, 255).astype(np.uint8)


def resize_image(img_arr: np.ndarray, target_size: tuple = (256, 256)) -> np.ndarray:
    """
    Resize image using high-quality Lanczos interpolation.
    """
    pil_img = Image.fromarray(img_arr)
    resized = pil_img.resize(target_size, resample=Image.Resampling.LANCZOS)
    return np.array(resized)
