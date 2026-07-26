from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


COLOR_MODES = ("grayscale", "rgb")
COLOR_ORDERS = ("rgb", "bgr")
NORMALIZATIONS = ("zero_one", "minus_one_one")


@dataclass(frozen=True, slots=True)
class VisionObservationConfig:
    width: int = 64
    height: int = 48
    color_mode: str = "grayscale"
    input_color_order: str = "rgb"
    frame_stack: int = 1
    include_delta: bool = False
    include_motion_mask: bool = False
    motion_threshold: float = 0.08
    normalization: str = "minus_one_one"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("vision width and height must be positive")
        if self.color_mode not in COLOR_MODES:
            raise ValueError(f"unsupported vision color mode {self.color_mode!r}")
        if self.input_color_order not in COLOR_ORDERS:
            raise ValueError(f"unsupported input color order {self.input_color_order!r}")
        if self.frame_stack <= 0:
            raise ValueError("vision frame_stack must be positive")
        if not 0.0 <= self.motion_threshold <= 1.0:
            raise ValueError("vision motion_threshold must be in [0, 1]")
        if self.normalization not in NORMALIZATIONS:
            raise ValueError(f"unsupported vision normalization {self.normalization!r}")

    @property
    def image_channels(self) -> int:
        return 1 if self.color_mode == "grayscale" else 3

    @property
    def channels(self) -> int:
        temporal = int(self.include_delta) + int(self.include_motion_mask)
        return self.image_channels * self.frame_stack + temporal

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.channels, self.height, self.width)

    @property
    def flat_dim(self) -> int:
        return self.channels * self.height * self.width

    def metadata(self) -> dict[str, object]:
        return {**asdict(self), "channels": self.channels, "shape": list(self.shape), "flat_dim": self.flat_dim}


class VisionObservationEncoder:
    def __init__(self, config: VisionObservationConfig) -> None:
        self.config = config
        self._history: deque[np.ndarray] = deque(maxlen=config.frame_stack)
        self._previous_gray: np.ndarray | None = None

    def reset(self) -> None:
        self._history.clear()
        self._previous_gray = None

    def encode(self, frame: np.ndarray) -> np.ndarray:
        image, gray = _prepare_frame(frame, self.config)
        previous_gray = self._previous_gray
        self._history.append(image)

        history = list(self._history)
        if len(history) < self.config.frame_stack:
            history = [history[0]] * (self.config.frame_stack - len(history)) + history
        channels = [_normalize_image(item, self.config.normalization) for item in history]

        if self.config.include_delta:
            delta = np.zeros_like(gray) if previous_gray is None else gray - previous_gray
            channels.append(delta[None, ...].astype(np.float32))
        if self.config.include_motion_mask:
            magnitude = np.zeros_like(gray) if previous_gray is None else np.abs(gray - previous_gray)
            channels.append((magnitude >= self.config.motion_threshold)[None, ...].astype(np.float32))

        self._previous_gray = gray
        encoded = np.concatenate(channels, axis=0).astype(np.float32, copy=False)
        if encoded.shape != self.config.shape:
            raise RuntimeError(f"vision encoder produced {encoded.shape}, expected {self.config.shape}")
        return np.ascontiguousarray(encoded)

    def encode_flat(self, frame: np.ndarray) -> np.ndarray:
        return self.encode(frame).reshape(-1)


class VisionObservationBatchEncoder:
    def __init__(self, config: VisionObservationConfig, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("vision batch_size must be positive")
        self.config = config
        self.encoders = tuple(VisionObservationEncoder(config) for _ in range(batch_size))

    @property
    def batch_size(self) -> int:
        return len(self.encoders)

    def reset(self) -> None:
        for encoder in self.encoders:
            encoder.reset()

    def encode(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        if len(frames) != self.batch_size:
            raise ValueError(f"expected {self.batch_size} vision frames, got {len(frames)}")
        return np.stack([encoder.encode_flat(frame) for encoder, frame in zip(self.encoders, frames, strict=True)])


def append_vision_observation(state: np.ndarray, vision: np.ndarray) -> np.ndarray:
    state_array = np.asarray(state, dtype=np.float32)
    vision_array = np.asarray(vision, dtype=np.float32)
    if state_array.ndim != vision_array.ndim:
        raise ValueError(f"state and vision ranks differ: {state_array.ndim} != {vision_array.ndim}")
    if state_array.ndim not in (1, 2):
        raise ValueError("state and vision observations must be vectors or batches of vectors")
    if state_array.ndim == 2 and state_array.shape[0] != vision_array.shape[0]:
        raise ValueError(f"state and vision batch sizes differ: {state_array.shape[0]} != {vision_array.shape[0]}")
    return np.concatenate((state_array, vision_array), axis=-1).astype(np.float32, copy=False)


def _prepare_frame(frame: np.ndarray, config: VisionObservationConfig) -> tuple[np.ndarray, np.ndarray]:
    source = _to_unit_float(frame)
    rgb = _as_rgb(source, config.input_color_order)
    gray = _rgb_to_gray(rgb)
    resized_gray = _resize_bilinear(gray, config.height, config.width)
    if config.color_mode == "grayscale":
        image = resized_gray[None, ...]
    else:
        resized_rgb = _resize_bilinear(rgb, config.height, config.width)
        image = np.moveaxis(resized_rgb, -1, 0)
    return image.astype(np.float32, copy=False), resized_gray.astype(np.float32, copy=False)


def _to_unit_float(frame: np.ndarray) -> np.ndarray:
    source = np.asarray(frame)
    if source.ndim not in (2, 3):
        raise ValueError(f"vision frame must have shape HxW or HxWxC, got {source.shape}")
    if source.size == 0 or not np.all(np.isfinite(source)):
        raise ValueError("vision frame is empty or contains non-finite values")
    values = source.astype(np.float32)
    minimum = float(values.min())
    maximum = float(values.max())
    if minimum < 0.0 or maximum > 255.0:
        raise ValueError(f"vision pixels must be in [0, 1] or [0, 255], got [{minimum}, {maximum}]")
    if maximum > 1.0:
        values *= 1.0 / 255.0
    return values


def _as_rgb(source: np.ndarray, color_order: str) -> np.ndarray:
    if source.ndim == 2:
        return np.repeat(source[..., None], 3, axis=2)
    if source.shape[2] == 1:
        return np.repeat(source, 3, axis=2)
    if source.shape[2] not in (3, 4):
        raise ValueError(f"vision frame must have 1, 3, or 4 channels, got {source.shape[2]}")
    rgb = source[..., :3]
    return rgb[..., ::-1] if color_order == "bgr" else rgb


def _rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def _resize_bilinear(image: np.ndarray, height: int, width: int) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    if (source_height, source_width) == (height, width):
        return image.copy()
    y = np.linspace(0.0, source_height - 1, height, dtype=np.float32)
    x = np.linspace(0.0, source_width - 1, width, dtype=np.float32)
    y0 = np.floor(y).astype(np.intp)
    x0 = np.floor(x).astype(np.intp)
    y1 = np.minimum(y0 + 1, source_height - 1)
    x1 = np.minimum(x0 + 1, source_width - 1)
    wy = (y - y0).reshape((height, 1) + (1,) * (image.ndim - 2))
    wx = (x - x0).reshape((1, width) + (1,) * (image.ndim - 2))
    top = image[y0][:, x0] * (1.0 - wx) + image[y0][:, x1] * wx
    bottom = image[y1][:, x0] * (1.0 - wx) + image[y1][:, x1] * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32)


def _normalize_image(image: np.ndarray, normalization: str) -> np.ndarray:
    if normalization == "minus_one_one":
        return image * 2.0 - 1.0
    return image
