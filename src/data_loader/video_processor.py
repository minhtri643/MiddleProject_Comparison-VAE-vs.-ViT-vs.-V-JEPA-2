"""
Video Preprocessing and Temporal Sampling Utilities.
Extracts uniform temporal clips and resizes frames for video representation models.
"""

from pathlib import Path
from typing import Tuple, Union
import cv2
import numpy as np
from PIL import Image, ImageOps


def load_and_sample_video(
    video_path: Union[str, Path],
    num_frames: int = 4,
    target_size: Tuple[int, int] = (256, 256)
) -> np.ndarray:
    """
    Reads a video file using OpenCV, uniformly samples `num_frames` across the duration,
    and center-crops each frame to `target_size`.
    
    Returns:
        np.ndarray of shape (num_frames, target_size[0], target_size[1], 3), uint8.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found at {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video contains 0 frames: {path}")

    # Determine frame indices to sample uniformly
    if total_frames < num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, endpoint=True).astype(int)

    sampled_frames = []
    current_frame_idx = 0

    for target_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            # Fallback if frame read fails: repeat last frame or black
            if len(sampled_frames) > 0:
                frame_rgb = sampled_frames[-1].copy()
            else:
                frame_rgb = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            cropped = ImageOps.fit(pil_img, target_size, method=Image.Resampling.LANCZOS)
            frame_rgb = np.array(cropped, dtype=np.uint8)

        sampled_frames.append(frame_rgb)

    cap.release()
    return np.stack(sampled_frames, axis=0)

