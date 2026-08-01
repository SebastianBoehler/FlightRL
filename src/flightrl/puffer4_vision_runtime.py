from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
import torch
from torch import nn

from .puffer4_vision_policy import (
    VISION_INTENT_DIM,
    FlightRLVisionEncoder,
    infer_vision_shape,
)


class _MinGRU(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, 3 * hidden_size, bias=False)]
        )

    @staticmethod
    def _g(values: torch.Tensor) -> torch.Tensor:
        return torch.where(values >= 0, values + 0.5, values.sigmoid())

    @staticmethod
    def _highway(
        inputs: torch.Tensor,
        recurrent: torch.Tensor,
        projection: torch.Tensor,
    ) -> torch.Tensor:
        gate = projection.sigmoid()
        return gate * recurrent + (1.0 - gate) * inputs

    def initial_state(self, batch_size: int) -> tuple[torch.Tensor]:
        return (torch.zeros(1, batch_size, self.hidden_size),)

    def forward_eval(
        self,
        inputs: torch.Tensor,
        state: tuple[torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor]]:
        recurrent_state = state[0]
        hidden, gate, projection = self.layers[0](inputs).chunk(3, dim=-1)
        recurrent = torch.lerp(
            recurrent_state[0],
            self._g(hidden),
            gate.sigmoid(),
        )
        output = self._highway(inputs, recurrent, projection)
        return output, (recurrent.unsqueeze(0),)


class _VisualDecoder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.decoder_logstd = nn.Parameter(torch.zeros(1, 4))
        self.decoder_mean = nn.Linear(hidden_size, 4)
        self.value_function = nn.Linear(hidden_size, 1)


class VisualPufferRuntime(nn.Module):
    """Minimal inference-only equivalent of the trained Puffer policy."""

    def __init__(self, observation_size: int, hidden_size: int) -> None:
        super().__init__()
        self.encoder = FlightRLVisionEncoder(observation_size, hidden_size)
        self.decoder = _VisualDecoder(hidden_size)
        self.network = _MinGRU(hidden_size)

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path) -> VisualPufferRuntime:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        observation_size = (
            int(state_dict["encoder.vision.1.weight"].shape[1])
            + VISION_INTENT_DIM
        )
        hidden_size = int(state_dict["encoder.fusion.0.weight"].shape[0])
        policy = cls(observation_size, hidden_size)
        policy.load_state_dict(state_dict, strict=True)
        policy.eval()
        return policy

    def initial_state(self, batch_size: int = 1) -> tuple[torch.Tensor]:
        return self.network.initial_state(batch_size)

    def forward_eval(
        self,
        observation: torch.Tensor,
        state: tuple[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor]]:
        encoded = self.encoder(observation)
        hidden, next_state = self.network.forward_eval(encoded, state)
        return (
            self.decoder.decoder_mean(hidden),
            self.decoder.value_function(hidden),
            next_state,
        )


class VisualObservationEncoder:
    """Real-frame implementation of the native gray4 visual contract."""

    def __init__(self, observation_size: int) -> None:
        self.width, self.height = infer_vision_shape(observation_size)
        self.previous: np.ndarray | None = None

    def reset(self) -> None:
        self.previous = None

    def encode(self, frame: np.ndarray, intent: np.ndarray) -> np.ndarray:
        values = np.asarray(frame, dtype=np.uint8)
        image = Image.fromarray(values).convert("L").resize(
            (self.width, self.height),
            Image.Resampling.BILINEAR,
        )
        current = np.asarray(image, dtype=np.float32)
        current = np.clip(17.0 * np.rint(current / 17.0), 0.0, 255.0)
        delta = (
            np.zeros_like(current)
            if self.previous is None
            else (current - self.previous) / 255.0
        )
        contrast = max(float(current.std()), 17.0)
        appearance = 0.5 * np.clip(
            (current - float(current.mean())) / contrast,
            -2.0,
            2.0,
        )
        motion = (np.abs(delta) >= 0.08).astype(np.float32)
        self.previous = current
        intent_values = np.asarray(intent, dtype=np.float32)
        if intent_values.shape != (VISION_INTENT_DIM,):
            raise ValueError(f"intent must have shape ({VISION_INTENT_DIM},)")
        return np.concatenate(
            (
                appearance.reshape(-1),
                delta.reshape(-1),
                motion.reshape(-1),
                intent_values,
            )
        ).astype(np.float32)


class VisualPufferShadow:
    """Runs the native visual policy without issuing hardware commands."""

    def __init__(self, checkpoint: str | Path) -> None:
        self.policy = VisualPufferRuntime.from_checkpoint(checkpoint)
        observation_size = self.policy.encoder.vision_dim + VISION_INTENT_DIM
        self.observation = VisualObservationEncoder(observation_size)
        self.state = self.policy.initial_state()

    def reset(self) -> None:
        self.observation.reset()
        self.state = self.policy.initial_state()

    @torch.no_grad()
    def step(self, frame: np.ndarray, intent: np.ndarray) -> dict[str, float | bool]:
        observation = self.observation.encode(frame, intent)
        pixels = self.observation.width * self.observation.height
        started = perf_counter()
        action, value, self.state = self.policy.forward_eval(
            torch.from_numpy(observation[None, :]),
            self.state,
        )
        inference_ms = 1_000.0 * (perf_counter() - started)
        bounded = action[0].clamp(-1.0, 1.0).numpy()
        return {
            "monitor_only": True,
            "controls_drone": False,
            "action_vx": float(bounded[0]),
            "action_vy": float(bounded[1]),
            "action_vz": float(bounded[2]),
            "action_yaw": float(bounded[3]),
            "value": float(value[0, 0]),
            "inference_ms": inference_ms,
            "input_contrast_std": float(observation[:pixels].std()),
            "input_delta_mae": float(
                np.abs(observation[pixels : 2 * pixels]).mean()
            ),
            "input_motion_fraction": float(
                observation[2 * pixels : 3 * pixels].mean()
            ),
        }
