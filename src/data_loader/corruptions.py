"""
Adverse Weather & Common Corruptions Simulation.
"""

from typing import Union
import cv2
import numpy as np
from PIL import Image


def apply_fog_corruption(img_array: np.ndarray, severity: int = 3) -> np.ndarray:
    """
    Simulates optical fog and atmospheric scattering based on Koschmieder's Law:
        I_fog(x) = I(x) * t(x) + L * (1 - t(x))
    where t(x) is transmission map and L is atmospheric airlight.

        """
    h, w = img_array.shape[:2]
    # Transmission coefficients based on severity
    beta_dict = {1: 0.05, 2: 0.10, 3: 0.18, 4: 0.28, 5: 0.40}
    beta = beta_dict.get(severity, 0.18)

    # Synthetic depth gradient (fog gets denser towards horizon/upper scene)
    y_coords = np.linspace(0.2, 1.0, h)[:, None]
    depth_map = np.tile(y_coords, (1, w))

    # Transmission map
    transmission = np.exp(-beta * (1.0 / (depth_map + 1e-4)))
    transmission = np.clip(transmission, 0.2, 0.95)[:, :, None]

    # Airlight (light gray atmospheric haze)
    airlight = np.array([210.0, 215.0, 220.0], dtype=np.float32)

    foggy = img_array.astype(np.float32) * transmission + airlight * (1.0 - transmission)
    # Slight Gaussian blur to simulate water droplet scattering
    foggy = cv2.GaussianBlur(foggy, (5, 5), 1.2)
    return np.clip(foggy, 0, 255).astype(np.uint8)


def corrupt_visual_input(visual_input: Union[Image.Image, np.ndarray], severity: int = 3):
    """
    Applies adverse weather corruption to either PIL.Image or (T, H, W, C) video array.
    """
    if isinstance(visual_input, Image.Image):
        arr = np.array(visual_input)
        corrupted_arr = apply_fog_corruption(arr, severity=severity)
        return Image.fromarray(corrupted_arr)
    elif isinstance(visual_input, np.ndarray):
        if visual_input.ndim == 3:
            return apply_fog_corruption(visual_input, severity=severity)
        elif visual_input.ndim == 4:
            # Video sequence: apply frame by frame
            corrupted_frames = [apply_fog_corruption(frame, severity=severity) for frame in visual_input]
            return np.stack(corrupted_frames, axis=0)
    else:
        raise TypeError("Unsupported visual input type for corruption.")

