from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _rendering_probe() -> tuple[bool, str]:
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
        with mujoco.Renderer(model, height=8, width=8):
            pass
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def is_mujoco_rendering_available() -> bool:
    return _rendering_probe()[0]


def require_mujoco_rendering() -> None:
    available, error = _rendering_probe()
    if not available:
        raise RuntimeError(
            "MuJoCo rendering backend is unavailable in this process: "
            f"{error}"
        )


def render_rgb(
    env,
    width: int,
    height: int,
    env_index: int,
    camera: str | None,
) -> np.ndarray:
    require_mujoco_rendering()
    with env.mujoco.Renderer(env.model, height=height, width=width) as renderer:
        renderer.update_scene(env.data[env_index], camera=camera)
        return renderer.render()


def render_aideck_gray(
    env,
    width: int,
    height: int,
    env_index: int,
) -> np.ndarray:
    rgb = render_rgb(env, width, height, env_index, "aideck").astype(np.float32)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return np.clip(gray, 0.0, 255.0).astype(np.uint8)


def render_aideck_gray4(
    env,
    width: int,
    height: int,
    env_index: int,
) -> np.ndarray:
    gray = render_aideck_gray(env, width, height, env_index)
    quantized = np.rint(gray.astype(np.float32) / 17.0) * 17.0
    return np.clip(quantized, 0.0, 255.0).astype(np.uint8)
