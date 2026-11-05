"""
VAE Native Decoder implementation for Latent-to-Pixel Reconstruction.

  High-Resolution Image Synthesis with Latent Diffusion Models.
"""

from typing import Optional
import numpy as np
import torch
from diffusers import AutoencoderKL


class VAEDecoder:
    """
    Decodes continuous spatial latent representations S_VAE in R^(32 x 32 x 4)
    back to pixel space R^(256 x 256 x 3) using SD-VAE's native convolution-deconvolution blocks.
    """
    def __init__(self, checkpoint: str = "stabilityai/sd-vae-ft-mse", device: Optional[str] = None):
        self.checkpoint = checkpoint
        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.dtype = torch.float32  # Float32 ensures high fidelity reconstruction
        self.model = None

    def load_model(self):
        if self.model is None:
            print(f"[VAEDecoder] Loading native VAE decoder from {self.checkpoint} on {self.device}...")
            self.model = AutoencoderKL.from_pretrained(self.checkpoint, torch_dtype=self.dtype)
            self.model.to(self.device).eval()

    def decode(self, latent_grid: np.ndarray, target_size=None) -> np.ndarray:
        """
        Args:
            latent_grid: np.ndarray of shape (32, 32, 4) or (1, 32, 32, 4)
            
        Returns:
            reconstructed_image: np.ndarray of shape (256, 256, 3), uint8 in [0, 255]
        """
        self.load_model()
        
        # Ensure shape (32, 32, 4)
        if latent_grid.ndim == 4 and latent_grid.shape[0] == 1:
            latent_grid = latent_grid[0]
            
        if latent_grid.shape != (32, 32, 4):
            raise ValueError(f"Expected latent grid of shape (32, 32, 4), got {latent_grid.shape}")

        # Permute (32, 32, 4) -> (1, 4, 32, 32)
        tensor = torch.from_numpy(latent_grid).permute(2, 0, 1).unsqueeze(0).to(
            device=self.device, dtype=self.dtype
        )

        with torch.inference_mode():
            # Native VAE decoder forward pass
            recon_tensor = self.model.decode(tensor).sample
            
            # Post-process: [-1, 1] -> [0, 255] uint8 (256, 256, 3)
            recon_arr = (recon_tensor[0].permute(1, 2, 0).clamp(-1.0, 1.0) + 1.0) * 127.5
            recon_uint8 = recon_arr.cpu().numpy().astype(np.uint8)

        return recon_uint8
