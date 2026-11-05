"""
VAE Encoder Wrapper for Pixel-Generative Latent Representation.

"""

import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL
from .base_encoder import BaseEncoder


class VAEEncoder(BaseEncoder):
    def __init__(self, checkpoint: str = "stabilityai/sd-vae-ft-mse", device: str = "auto", use_fp16: bool = True):
        super().__init__(name="VAE", checkpoint=checkpoint, device=device, use_fp16=use_fp16)

    def load_model(self):
        if self.model is None:
            # VAE typically runs in float32 for numerical stability in latent distribution calculation
            self.model = AutoencoderKL.from_pretrained(self.checkpoint, torch_dtype=self.dtype)
            self.model.to(self.device).eval()

    def encode(self, visual_input) -> np.ndarray:
        """
        Input: PIL.Image (256x256) or np.ndarray (H, W, 3) or sequence of frames.
        Returns: Spatial latent grid of shape (32, 32, 4).
        """
        self.load_model()
        if isinstance(visual_input, Image.Image):
            img_arr = np.array(visual_input, copy=True)
        elif isinstance(visual_input, np.ndarray):
            # If video clip (T, H, W, C), take middle frame or average
            if visual_input.ndim == 4:
                mid_idx = visual_input.shape[0] // 2
                img_arr = visual_input[mid_idx]
            else:
                img_arr = visual_input
        else:
            raise TypeError("Unsupported visual input type for VAE.")

        # Convert to tensor [-1, 1] range: (1, 3, 256, 256)
        tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).float()
        tensor = tensor.to(device=self.device, dtype=self.dtype) / 127.5 - 1.0

        with torch.inference_mode():
            # Extract latent mode: (1, 4, 32, 32)
            latent = self.model.encode(tensor).latent_dist.mode()
            # Permute to (32, 32, 4)
            grid = latent[0].permute(1, 2, 0).float().cpu().numpy().copy()

        return self.validate_grid(grid, "VAE")

