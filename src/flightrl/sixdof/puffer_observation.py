from __future__ import annotations

from .env import ACTION_DIM


def scale_previous_action_observation(observations, scale: float):
    if float(scale) == 1.0:
        return observations
    output = observations.clone() if hasattr(observations, "clone") else observations.copy()
    output[..., -ACTION_DIM:] *= float(scale)
    return output
