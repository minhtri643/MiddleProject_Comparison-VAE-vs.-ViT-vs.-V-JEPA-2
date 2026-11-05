"""
Image Preprocessing Utilities.
Handles EXIF orientation, RGB conversion, and high-fidelity square cropping.
"""

from pathlib import Path
from typing import Tuple, Union
import numpy as np
from PIL import Image, ImageOps


def load_and_preprocess_image(image_path: Union[str, Path], target_size: Tuple[int, int] = (256, 256)) -> Image.Image:
    """
    Loads an image from path, corrects EXIF orientation, converts to RGB,
    and crops to a centered square of target_size using Lanczos resampling.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at {path}")

    with Image.open(path) as src:
        image = ImageOps.exif_transpose(src).convert("RGB")
        processed_image = ImageOps.fit(image, target_size, method=Image.Resampling.LANCZOS)
    return processed_image


def image_to_numpy(image: Image.Image) -> np.ndarray:
    """Converts PIL.Image to numpy array uint8 (H, W, 3)."""
    return np.array(image, dtype=np.uint8)

