from __future__ import annotations

import torch

from flightrl.puffer4_door_evidence import DOOR_EVIDENCE_DIM
from flightrl.puffer4_door_observation import DOOR_PHASE_DIM, DOOR_SENSOR_DIM
from flightrl.puffer4_door_policy import (
    DOOR_OBS_DIM,
    DOOR_PIXELS,
    DOOR_POLICY_OBS_DIM,
)
from flightrl.puffer4_door_self_mask import door_self_mask


CAMERA_STOP = 3 * DOOR_PIXELS
SENSOR_STOP = CAMERA_STOP + DOOR_SENSOR_DIM
PHASE_STOP = SENSOR_STOP + DOOR_PHASE_DIM
EVIDENCE_STOP = PHASE_STOP + DOOR_EVIDENCE_DIM
CAMERA_BUNDLE_SIZE = CAMERA_STOP + DOOR_POLICY_OBS_DIM - SENSOR_STOP


class DoorPixelNoiseTransform:
    """Apply isolated temporal pixel noise to the actor's camera channels."""

    def __init__(
        self,
        *,
        agent_count: int,
        seed: int,
        sigma_uint8: float = 8.5,
        quantization_step_uint8: float = 17.0,
    ) -> None:
        if agent_count <= 0:
            raise ValueError("pixel-noise challenge requires at least one agent")
        if sigma_uint8 <= 0.0 or quantization_step_uint8 <= 0.0:
            raise ValueError("pixel-noise scales must be positive")
        self.agent_count = agent_count
        self.seed = seed
        self.sigma_uint8 = sigma_uint8
        self.quantization_step_uint8 = quantization_step_uint8
        self._generator: torch.Generator | None = None
        self._previous: torch.Tensor | None = None
        self._seen: torch.Tensor | None = None
        self._mask = torch.as_tensor(
            door_self_mask(48, 64).reshape(-1).copy(),
            dtype=torch.bool,
        )

    def transform(self, observations: torch.Tensor) -> torch.Tensor:
        self._validate(observations)
        if self._generator is None:
            self._initialize(observations)
        assert self._generator is not None
        assert self._previous is not None
        assert self._seen is not None

        current = observations[:, :DOOR_PIXELS]
        noise = torch.randn(
            current.shape,
            generator=self._generator,
            dtype=current.dtype,
            device=current.device,
        )
        noisy_uint8 = current * 255.0 + noise * self.sigma_uint8
        quantized = (
            torch.round(noisy_uint8 / self.quantization_step_uint8)
            * self.quantization_step_uint8
        ).clamp(0.0, 255.0) / 255.0
        mask = self._mask.to(device=current.device)
        quantized[:, mask] = current[:, mask]

        delta = quantized - self._previous
        delta[~self._seen] = 0.0
        transformed = observations.clone()
        transformed[:, :DOOR_PIXELS] = quantized
        transformed[:, DOOR_PIXELS : 2 * DOOR_PIXELS] = delta
        transformed[:, 2 * DOOR_PIXELS:CAMERA_STOP] = (
            delta.abs() >= 0.08
        ).to(dtype=observations.dtype)
        self._previous.copy_(quantized)
        self._seen.fill_(True)
        return transformed

    def clear(self, terminals: torch.Tensor) -> None:
        _validate_terminals(terminals, self.agent_count)
        if self._previous is None:
            return
        assert self._seen is not None
        mask = terminals > 0.0
        self._previous[mask] = 0.0
        self._seen[mask] = False

    def mechanism_report(self) -> dict:
        return {
            "single_intervention": "pixel_noise",
            "seed": self.seed,
            "sigma_uint8": self.sigma_uint8,
            "quantization_step_uint8": self.quantization_step_uint8,
            "self_mask_preserved": True,
            "signed_delta_and_motion_recomputed": True,
            "terminal_cleared": True,
            "detector_evidence_recomputed": False,
            "limitation": (
                "Tests actor raster robustness while simulated detector "
                "phase/evidence remain clean."
            ),
        }

    def _initialize(self, observations: torch.Tensor) -> None:
        self._generator = torch.Generator(device=observations.device)
        self._generator.manual_seed(self.seed)
        self._previous = torch.zeros(
            (self.agent_count, DOOR_PIXELS),
            dtype=observations.dtype,
            device=observations.device,
        )
        self._seen = torch.zeros(
            self.agent_count,
            dtype=torch.bool,
            device=observations.device,
        )

    def _validate(self, observations: torch.Tensor) -> None:
        _validate_observations(observations, self.agent_count)
        if self._previous is not None and (
            self._previous.device != observations.device
            or self._previous.dtype != observations.dtype
        ):
            raise ValueError("pixel-noise observation storage changed")


class DoorCameraLatencyTransform:
    """Delay camera-derived policy channels while keeping telemetry current."""

    def __init__(
        self,
        *,
        agent_count: int,
        delay_steps: int,
        control_dt_s: float,
        maximum_evidence_age_s: float,
    ) -> None:
        if agent_count <= 0 or delay_steps <= 0:
            raise ValueError("camera latency requires agents and positive delay")
        if control_dt_s <= 0.0 or maximum_evidence_age_s <= 0.0:
            raise ValueError("camera latency time scales must be positive")
        self.agent_count = agent_count
        self.delay_steps = delay_steps
        self.control_dt_s = control_dt_s
        self.maximum_evidence_age_s = maximum_evidence_age_s
        self._history: torch.Tensor | None = None
        self._counts: torch.Tensor | None = None
        self._write: torch.Tensor | None = None

    def transform(self, observations: torch.Tensor) -> torch.Tensor:
        self._validate(observations)
        if self._history is None:
            self._initialize(observations)
        assert self._history is not None
        assert self._counts is not None
        assert self._write is not None

        bundle = torch.cat(
            (
                observations[:, :CAMERA_STOP],
                observations[:, SENSOR_STOP:DOOR_POLICY_OBS_DIM],
            ),
            dim=1,
        )
        transformed = observations.clone()
        full = self._counts >= self.delay_steps
        warmup = ~full
        transformed[warmup, :CAMERA_STOP] = 0.0
        transformed[warmup, SENSOR_STOP:EVIDENCE_STOP] = 0.0
        transformed[warmup, SENSOR_STOP] = 1.0
        transformed[warmup, EVIDENCE_STOP - 1] = 1.0

        indices = torch.arange(self.agent_count, device=observations.device)
        ready = indices[full]
        if ready.numel() > 0:
            historical = self._history[ready, self._write[ready]]
            transformed[ready, :CAMERA_STOP] = historical[:, :CAMERA_STOP]
            transformed[ready, SENSOR_STOP:DOOR_POLICY_OBS_DIM] = historical[
                :, CAMERA_STOP:
            ]
            self._advance_evidence_age(transformed, ready)

        self._history[indices, self._write] = bundle
        self._write = (self._write + 1) % self.delay_steps
        self._counts = torch.clamp(self._counts + 1, max=self.delay_steps)
        return transformed

    def clear(self, terminals: torch.Tensor) -> None:
        _validate_terminals(terminals, self.agent_count)
        if self._history is None:
            return
        assert self._counts is not None
        assert self._write is not None
        mask = terminals > 0.0
        self._history[mask] = 0.0
        self._counts[mask] = 0
        self._write[mask] = 0

    def mechanism_report(self) -> dict:
        return {
            "single_intervention": "camera_latency",
            "delay_steps": self.delay_steps,
            "control_dt_s": self.control_dt_s,
            "delay_ms": self.delay_steps * self.control_dt_s * 1_000.0,
            "maximum_evidence_age_s": self.maximum_evidence_age_s,
            "camera_bundle_delayed_together": True,
            "current_sensors_and_previous_action_preserved": True,
            "terminal_cleared": True,
            "limitation": (
                "Fixed step-quantized delay is additional to the native "
                "detector sample-and-hold and does not model jitter or drops."
            ),
        }

    def _advance_evidence_age(
        self,
        transformed: torch.Tensor,
        ready: torch.Tensor,
    ) -> None:
        age_index = EVIDENCE_STOP - 1
        increment = (
            self.delay_steps
            * self.control_dt_s
            / self.maximum_evidence_age_s
        )
        transformed[ready, age_index] = (
            transformed[ready, age_index] + increment
        ).clamp(max=1.0)
        stale = transformed[ready, age_index] >= 1.0
        stale_indices = ready[stale]
        if stale_indices.numel() == 0:
            return
        phases = transformed[stale_indices, SENSOR_STOP:PHASE_STOP]
        target_seen = phases[:, 1:].sum(dim=1) > 0.0
        transformed[stale_indices, SENSOR_STOP:PHASE_STOP] = 0.0
        transformed[stale_indices, SENSOR_STOP] = (~target_seen).to(
            transformed.dtype
        )
        transformed[stale_indices, PHASE_STOP - 1] = target_seen.to(
            transformed.dtype
        )
        transformed[stale_indices, PHASE_STOP:EVIDENCE_STOP] = 0.0
        transformed[stale_indices, age_index] = 1.0

    def _initialize(self, observations: torch.Tensor) -> None:
        self._history = torch.zeros(
            (self.agent_count, self.delay_steps, CAMERA_BUNDLE_SIZE),
            dtype=observations.dtype,
            device=observations.device,
        )
        self._counts = torch.zeros(
            self.agent_count,
            dtype=torch.long,
            device=observations.device,
        )
        self._write = torch.zeros_like(self._counts)

    def _validate(self, observations: torch.Tensor) -> None:
        _validate_observations(observations, self.agent_count)
        if self._history is not None and (
            self._history.device != observations.device
            or self._history.dtype != observations.dtype
        ):
            raise ValueError("camera-latency observation storage changed")


def _validate_observations(
    observations: torch.Tensor,
    agent_count: int,
) -> None:
    if observations.shape != (agent_count, DOOR_OBS_DIM):
        raise ValueError("observation shape does not match door contract")
    if not observations.is_floating_point():
        raise ValueError("door challenge observations must be floating point")


def _validate_terminals(terminals: torch.Tensor, agent_count: int) -> None:
    if terminals.shape != (agent_count,):
        raise ValueError("terminal mask does not match door challenge agents")
