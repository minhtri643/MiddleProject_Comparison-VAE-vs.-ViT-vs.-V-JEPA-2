"""
Abstract Base Encoder Class for Visual Representation Models.

Reference Literature:
"""

from abc import ABC, abstractmethod
import gc
from typing import Tuple, Dict, Any
import numpy as np
import torch


class BaseEncoder(ABC):
    def __init__(self, name: str, checkpoint: str, device: str = "auto", use_fp16: bool = True):
        self.name = name
        self.checkpoint = checkpoint
        self.use_fp16 = use_fp16
        self.device = self._resolve_device(device)
        self.dtype = torch.float16 if (self.device.type == "cuda" and self.use_fp16 and name != "VAE") else torch.float32
        self.model = None

    def _resolve_device(self, requested_device: str) -> torch.device:
        if requested_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif requested_device in {"cuda", "cpu"}:
            if requested_device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available on this machine.")
            return torch.device(requested_device)
        else:
            raise ValueError(f"Unknown device specification: {requested_device}")

    @abstractmethod
    def load_model(self):
        """Loads model weights into specified device and evaluation mode."""
        pass

    @abstractmethod
    def encode(self, visual_input: Any) -> np.ndarray:
        """
        Processes visual input and returns spatial patch grid features.
        Returns:
            np.ndarray of shape (H, W, D)
        """
        pass

    def release_memory(self):
        """Releases model weights and flushes GPU cache."""
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def validate_grid(grid: np.ndarray, name: str) -> np.ndarray:
        """Validates that grid has shape (H, W, D) with finite values."""
        arr = np.asarray(grid, dtype=np.float32)
        if arr.ndim != 3 or min(arr.shape[:2]) < 2 or arr.shape[-1] < 2:
            raise ValueError(f"{name}: Expected grid (H, W, D) with H,W>=2, got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name}: Grid contains non-finite values (NaN or Inf).")
        return arr

