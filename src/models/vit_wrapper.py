"""
Vision Transformer (ViT) Wrapper for Supervised Feature Representation.

"""

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, ViTModel
from .base_encoder import BaseEncoder


class ViTEncoder(BaseEncoder):
    def __init__(self, checkpoint: str = "google/vit-base-patch16-224", device: str = "auto", use_fp16: bool = True):
        super().__init__(name="ViT", checkpoint=checkpoint, device=device, use_fp16=use_fp16)
        self.processor = None

    def load_model(self):
        if self.model is None:
            self.processor = AutoImageProcessor.from_pretrained(self.checkpoint)
            self.model = ViTModel.from_pretrained(
                self.checkpoint,
                add_pooling_layer=False,
                torch_dtype=self.dtype,
                attn_implementation="sdpa" if hasattr(torch.nn.functional, "scaled_dot_product_attention") else "eager"
            )
            self.model.to(self.device).eval()

    def encode(self, visual_input) -> np.ndarray:
        """
        Input: PIL.Image or np.ndarray.
        Returns: Spatial patch tokens reshaped to (14, 14, 768).
        """
        self.load_model()
        if isinstance(visual_input, np.ndarray) and visual_input.ndim == 4:
            # If video clip, take middle frame
            mid_idx = visual_input.shape[0] // 2
            img_to_process = Image.fromarray(visual_input[mid_idx])
        elif isinstance(visual_input, np.ndarray):
            img_to_process = Image.fromarray(visual_input)
        else:
            img_to_process = visual_input

        inputs = self.processor(images=img_to_process, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.dtype)

        with torch.inference_mode():
            outputs = self.model(pixel_values=pixel_values)
            # Remove [CLS] token at index 0: shape (1, 196, 768) -> (196, 768)
            tokens = outputs.last_hidden_state[0, 1:]
            patch = int(self.model.config.patch_size)
            h = pixel_values.shape[-2] // patch
            w = pixel_values.shape[-1] // patch
            grid = tokens.float().cpu().numpy().reshape(h, w, -1).copy()

        return self.validate_grid(grid, "ViT")

