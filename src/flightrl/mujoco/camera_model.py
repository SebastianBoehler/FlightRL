from __future__ import annotations

import numpy as np

from .rendering import _quantize_gray4_high_nibble


def sample_gray4_camera_parameters(
    rng: np.random.Generator,
    mask: np.ndarray,
    target_means: np.ndarray,
    gammas: np.ndarray,
) -> None:
    count = int(np.sum(mask))
    target_means[mask] = rng.uniform(35.0, 90.0, size=count)
    gammas[mask] = rng.uniform(0.8, 1.2, size=count)


def randomize_gray4_frame(
    gray: np.ndarray,
    *,
    target_mean: float,
    gamma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    normalized = np.clip(gray / 255.0, 0.0, 1.0) ** gamma
    current_mean = max(float(normalized.mean() * 255.0), 1.0)
    adjusted = normalized * (target_mean / current_mean)
    noisy = adjusted * 255.0 + rng.normal(0.0, 2.0, size=gray.shape)
    return _quantize_gray4_high_nibble(
        np.clip(noisy, 0.0, 255.0).astype(np.uint8)
    )
