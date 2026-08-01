from __future__ import annotations

from functools import lru_cache
from typing import Protocol

import numpy as np


class _Grounder(Protocol):
    def infer(
        self,
        pixels: np.ndarray,
        prompt: str,
        *,
        frame_index: int,
        frame_host_time_s: float,
    ): ...


@lru_cache(maxsize=8)
def door_self_mask(height: int, width: int) -> np.ndarray:
    if height <= 0 or width <= 0:
        raise ValueError("door self-mask dimensions must be positive")
    y, x = np.ogrid[:height, :width]
    xn = (x + 0.5) / width
    yn = (y + 0.5) / height
    left = ((xn < 0.39) & (yn < 0.24)) | ((xn < 0.27) & (yn < 0.42))
    right = ((xn > 0.67) & (yn < 0.24)) | ((xn > 0.75) & (yn < 0.40))
    mask = left | right
    mask.setflags(write=False)
    return mask


def apply_door_self_mask(frame: np.ndarray) -> np.ndarray:
    pixels = np.asarray(frame)
    if pixels.ndim not in (2, 3):
        raise ValueError("door frames must be HxW or HxWxC")
    masked = pixels.copy()
    mask = door_self_mask(masked.shape[0], masked.shape[1])
    fill = np.mean(masked, axis=(0, 1)).astype(masked.dtype)
    masked[mask] = fill
    return masked


class DoorSelfMaskedGrounder:
    """Apply the calibrated airframe mask before host semantic inference."""

    def __init__(self, grounder: _Grounder) -> None:
        self.grounder = grounder

    def infer(
        self,
        pixels: np.ndarray,
        prompt: str,
        *,
        frame_index: int,
        frame_host_time_s: float,
    ):
        return self.grounder.infer(
            apply_door_self_mask(pixels),
            prompt,
            frame_index=frame_index,
            frame_host_time_s=frame_host_time_s,
        )
