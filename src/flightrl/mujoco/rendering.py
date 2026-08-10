from __future__ import annotations

from functools import lru_cache

import numpy as np

from .camera_contract import AIDECK_SOURCE_HEIGHT, AIDECK_SOURCE_WIDTH

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


def _quantize_gray4_high_nibble(gray: np.ndarray) -> np.ndarray:
    pixels = np.asarray(gray, dtype=np.uint8)
    return ((pixels >> 4) * 17).astype(np.uint8)


def _gap8_resize_gray4(
    source: np.ndarray,
    *,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    pixels = np.asarray(source, dtype=np.uint8)
    if pixels.ndim != 2:
        raise ValueError("GAP8 gray4 source must be a two-dimensional grayscale frame")
    if output_width <= 0 or output_height <= 0:
        raise ValueError("GAP8 gray4 output dimensions must be positive")
    x_map = np.arange(output_width, dtype=np.int64) * pixels.shape[1] // output_width
    y_map = np.arange(output_height, dtype=np.int64) * pixels.shape[0] // output_height
    return _quantize_gray4_high_nibble(pixels[np.ix_(y_map, x_map)])


def render_aideck_gray4(
    env,
    width: int,
    height: int,
    env_index: int,
) -> np.ndarray:
    source = render_aideck_gray(
        env,
        AIDECK_SOURCE_WIDTH,
        AIDECK_SOURCE_HEIGHT,
        env_index,
    )
    return _gap8_resize_gray4(
        source,
        output_width=width,
        output_height=height,
    )
