from __future__ import annotations

import random

import torch

from flightrl.puffer4_door_observation import DOOR_SENSOR_DIM
from flightrl.puffer4_door_policy import (
    DOOR_OBS_DIM,
    DOOR_PIXELS,
    DOOR_POLICY_OBS_DIM,
)


TEMPORAL_ORDER_ABLATION_SEED = 20_260_734
TEMPORAL_HISTORY_DEPTH = 4
_CAMERA_STOP = 3 * DOOR_PIXELS
_SENSOR_STOP = _CAMERA_STOP + DOOR_SENSOR_DIM
_BUNDLE_SIZE = _CAMERA_STOP + DOOR_POLICY_OBS_DIM - _SENSOR_STOP


class DoorTemporalOrderScrambler:
    """Causal, same-agent permutation of past visual observation bundles."""

    def __init__(self, *, agent_count: int, seed: int) -> None:
        if agent_count <= 0:
            raise ValueError("temporal ablation requires at least one agent")
        self.agent_count = agent_count
        self.seed = seed
        self.past_lag_permutation = tuple(
            random.Random(seed).sample(
                range(1, TEMPORAL_HISTORY_DEPTH + 1),
                TEMPORAL_HISTORY_DEPTH,
            )
        )
        monotonic = (
            tuple(range(1, TEMPORAL_HISTORY_DEPTH + 1)),
            tuple(range(TEMPORAL_HISTORY_DEPTH, 0, -1)),
        )
        if self.past_lag_permutation in monotonic:
            raise ValueError("temporal ablation seed produced a trivial order")
        self._history: torch.Tensor | None = None
        self._counts: torch.Tensor | None = None
        self._write: torch.Tensor | None = None
        self._cycle: torch.Tensor | None = None
        self._lags: torch.Tensor | None = None

    def transform(self, observations: torch.Tensor) -> torch.Tensor:
        self._validate_observations(observations)
        if self._history is None:
            self._initialize(observations)
        assert self._history is not None
        assert self._counts is not None
        assert self._write is not None
        assert self._cycle is not None
        assert self._lags is not None

        current_bundle = torch.cat(
            (
                observations[:, :_CAMERA_STOP],
                observations[:, _SENSOR_STOP:DOOR_POLICY_OBS_DIM],
            ),
            dim=1,
        )
        transformed = observations.clone()
        full = self._counts >= TEMPORAL_HISTORY_DEPTH
        full_indices = torch.nonzero(full, as_tuple=False).flatten()
        if full_indices.numel() > 0:
            lag = self._lags[self._cycle % TEMPORAL_HISTORY_DEPTH]
            slots = (self._write - lag) % TEMPORAL_HISTORY_DEPTH
            historical = self._history[
                full_indices,
                slots[full_indices],
            ]
            transformed[full_indices, :_CAMERA_STOP] = historical[
                :, :_CAMERA_STOP
            ]
            transformed[
                full_indices,
                _SENSOR_STOP:DOOR_POLICY_OBS_DIM,
            ] = historical[:, _CAMERA_STOP:]

        indices = torch.arange(self.agent_count, device=observations.device)
        self._history[indices, self._write] = current_bundle
        self._write = (self._write + 1) % TEMPORAL_HISTORY_DEPTH
        self._counts = torch.clamp(self._counts + 1, max=TEMPORAL_HISTORY_DEPTH)
        self._cycle = self._cycle + full.to(dtype=torch.long)
        return transformed

    def clear(self, terminals: torch.Tensor) -> None:
        if self._history is None:
            return
        if terminals.shape != (self.agent_count,):
            raise ValueError("terminal mask does not match temporal agents")
        mask = terminals > 0.0
        self._history[mask] = 0.0
        self._counts[mask] = 0
        self._write[mask] = 0
        self._cycle[mask] = 0

    def mechanism_report(self) -> dict:
        return {
            "seed": self.seed,
            "history_depth": TEMPORAL_HISTORY_DEPTH,
            "past_lag_permutation": list(self.past_lag_permutation),
            "same_agent_only": True,
            "terminal_cleared": True,
            "ordered_bootstrap_steps_per_episode": TEMPORAL_HISTORY_DEPTH,
            "scrambled_together": [
                "current_gray4",
                "signed_delta",
                "motion",
                "phase",
                "detector_evidence",
            ],
            "preserved_current": [
                "all_sensors",
                "executed_previous_action",
                "privileged_teacher_actor_invisible",
            ],
        }

    def _initialize(self, observations: torch.Tensor) -> None:
        device = observations.device
        self._history = torch.zeros(
            (
                self.agent_count,
                TEMPORAL_HISTORY_DEPTH,
                _BUNDLE_SIZE,
            ),
            dtype=observations.dtype,
            device=device,
        )
        self._counts = torch.zeros(
            self.agent_count,
            dtype=torch.long,
            device=device,
        )
        self._write = torch.zeros_like(self._counts)
        self._cycle = torch.zeros_like(self._counts)
        self._lags = torch.tensor(
            self.past_lag_permutation,
            dtype=torch.long,
            device=device,
        )

    def _validate_observations(self, observations: torch.Tensor) -> None:
        if observations.shape != (self.agent_count, DOOR_OBS_DIM):
            raise ValueError("observation shape does not match temporal contract")
        if not observations.is_floating_point():
            raise ValueError("temporal observations must be floating point")
        if self._history is not None and (
            observations.device != self._history.device
            or observations.dtype != self._history.dtype
        ):
            raise ValueError("temporal observation storage changed")


def build_temporal_order_ablation(
    carried_ordered: dict,
    scrambled: dict,
    scrambler: DoorTemporalOrderScrambler,
) -> dict:
    keys = ("success_rate", "outside_fov_success_rate", "collision_rate")
    return {
        "label": "causal_visual_temporal_order_scramble",
        "actual_temporal_order_ablation": True,
        "condition": {
            "camera": "full",
            "recurrent_mode": "carried",
            "yaw_cap": "unchanged",
        },
        "mechanism": scrambler.mechanism_report(),
        "metrics": scrambled,
        "delta_vs_carried_ordered": {
            key: scrambled.get(key, 0.0) - carried_ordered.get(key, 0.0)
            for key in keys
        },
        "causal_online_limitations": [
            (
                "Only already observed frames are used; arbitrary permutations "
                "that require future frames are impossible online."
            ),
            (
                "The cyclic lag permutation changes visual age and may repeat "
                "or omit source frames, so order cannot be isolated from age "
                "and coverage as cleanly as in an offline sequence test."
            ),
            (
                "The first four steps after each terminal remain ordered to "
                "avoid cross-episode history leakage."
            ),
            (
                "Current proprioception and executed previous action remain "
                "aligned, so this measures visual temporal coherence rather "
                "than a complete observation-order shuffle."
            ),
        ],
    }
