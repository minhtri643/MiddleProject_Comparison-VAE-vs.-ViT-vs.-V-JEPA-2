"""
V-JEPA 2 Wrapper for Video Joint Embedding Predictive Architecture.

"""

import numpy as np
import torch
from PIL import Image
from transformers import AutoVideoProcessor, AutoModel
from .base_encoder import BaseEncoder


class VJEPAEncoder(BaseEncoder):
    def __init__(self, checkpoint: str = "facebook/vjepa2-vitl-fpc64-256", device: str = "auto", use_fp16: bool = True, frames: int = 4):
        super().__init__(name="V-JEPA 2", checkpoint=checkpoint, device=device, use_fp16=use_fp16)
        if frames < 2 or frames % 2 != 0:
            raise ValueError("frames must be an even number >= 2 (tubelet_size = 2).")
        self.frames = frames
        self.processor = None

    def load_model(self):
        if self.model is None:
            self.processor = AutoVideoProcessor.from_pretrained(self.checkpoint)
            self.model = AutoModel.from_pretrained(
                self.checkpoint,
                torch_dtype=self.dtype,
                attn_implementation="sdpa" if hasattr(torch.nn.functional, "scaled_dot_product_attention") else "eager",
                low_cpu_mem_usage=True
            )
            self.model.to(self.device).eval()

    def encode(self, visual_input) -> np.ndarray:
        """
        Input: PIL.Image (replicated across temporal dimension) or np.ndarray (T, H, W, C).
        Returns: Spatial grid (16, 16, 1024) via temporal pooling over spatio-temporal tubelets.
        """
        self.load_model()
        if isinstance(visual_input, Image.Image):
            frame_arr = np.array(visual_input, copy=True)
            clip = np.repeat(frame_arr[None], self.frames, axis=0)
        elif isinstance(visual_input, np.ndarray):
            if visual_input.ndim == 3:  # (H, W, C)
                clip = np.repeat(visual_input[None], self.frames, axis=0)
            elif visual_input.ndim == 4:  # (T, H, W, C)
                # Resample or take self.frames
                if visual_input.shape[0] != self.frames:
                    indices = np.linspace(0, visual_input.shape[0] - 1, self.frames).astype(int)
                    clip = visual_input[indices]
                else:
                    clip = visual_input
            else:
                raise ValueError("Invalid numpy array dimensions for V-JEPA.")
        else:
            raise TypeError("Unsupported visual input type for V-JEPA.")

        inputs = self.processor(videos=list(clip), do_sample_frames=False, return_tensors="pt")
        pixels = inputs["pixel_values_videos"].to(device=self.device, dtype=self.dtype)

        with torch.inference_mode():
            outputs = self.model(pixel_values_videos=pixels, skip_predictor=True)
            tokens = outputs.last_hidden_state[0]

            patch = self.model.config.patch_size
            ph, pw = patch if isinstance(patch, (list, tuple)) else (patch, patch)
            tubelet = getattr(self.model.config, "tubelet_size", 2)
            t = self.frames // tubelet
            h = pixels.shape[-2] // ph
            w = pixels.shape[-1] // pw

            if len(tokens) != t * h * w:
                raise ValueError(f"Token count mismatch in V-JEPA: {len(tokens)} vs {t * h * w}")

            # Shape: (t, h, w, D) -> average over time tubelets -> (h, w, D)
            grid = tokens.float().cpu().numpy().reshape(t, h, w, -1).mean(axis=0).copy()

        return self.validate_grid(grid, "V-JEPA 2")

