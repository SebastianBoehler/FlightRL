from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

from flightrl.puffer4_door_policy import (
    DOOR_OBS_DIM,
    DOOR_PIXELS,
    DOOR_POLICY_OBS_DIM,
)
from flightrl.puffer4_door_challenges import DoorCameraLatencyTransform
from flightrl.puffer4_door_promotion_eval import (
    evaluate_promotion_door_policy,
)
from flightrl.puffer4_door_observation import DOOR_SENSOR_DIM
from flightrl.puffer4_door_temporal_ablation import (
    TEMPORAL_ORDER_ABLATION_SEED,
    DoorTemporalOrderScrambler,
    build_temporal_order_ablation,
)


CAMERA_STOP = 3 * DOOR_PIXELS
SENSOR_STOP = CAMERA_STOP + DOOR_SENSOR_DIM


def _observations(first: float, second: float) -> torch.Tensor:
    result = torch.zeros((2, DOOR_OBS_DIM))
    for agent, value in enumerate((first, second)):
        result[agent, :CAMERA_STOP] = value
        result[agent, CAMERA_STOP:SENSOR_STOP] = value + 1_000.0
        result[agent, SENSOR_STOP:DOOR_POLICY_OBS_DIM] = value + 2_000.0
        result[agent, DOOR_POLICY_OBS_DIM:] = value + 3_000.0
    return result


class _TemporalVec:
    total_agents = 1
    obs_size = DOOR_OBS_DIM

    def __init__(self) -> None:
        self.observations = torch.zeros((1, DOOR_OBS_DIM))
        self.terminals = torch.zeros(1)
        self.obs_ptr = self.observations.data_ptr()
        self.terminals_ptr = self.terminals.data_ptr()
        self.steps = 0
        self._fill(0.0)

    def _fill(self, value: float) -> None:
        self.observations[:, :CAMERA_STOP] = value
        self.observations[:, CAMERA_STOP:SENSOR_STOP] = value + 1_000.0
        self.observations[:, SENSOR_STOP:DOOR_POLICY_OBS_DIM] = value + 2_000.0
        self.observations[:, DOOR_POLICY_OBS_DIM:] = value + 3_000.0

    def reset(self) -> None:
        self.steps = 0
        self.terminals.zero_()
        self._fill(0.0)

    def cpu_step(self, _actions_ptr: int) -> None:
        self.steps += 1
        self._fill(float(self.steps))

    def log(self) -> dict[str, float]:
        return {
            "n": 1.0,
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "outside_fov_episode_fraction": 0.0,
            "outside_fov_success_fraction": 0.0,
        }

    def close(self) -> None:
        return None


class _TemporalPuffer:
    def __init__(self) -> None:
        self.vec = _TemporalVec()
        self._C = SimpleNamespace(
            create_vec=lambda _args, _gpu: self.vec,
            gpu=0,
        )

    def _cpu_tensor(
        self,
        pointer: int,
        _shape: tuple[int, ...],
        _dtype: torch.dtype,
    ) -> torch.Tensor:
        return {
            self.vec.obs_ptr: self.vec.observations,
            self.vec.terminals_ptr: self.vec.terminals,
        }[pointer]


class _RecordingPolicy:
    def __init__(self) -> None:
        self.observations: list[torch.Tensor] = []

    def initial_state(
        self,
        batch_size: int,
        device: str,
    ) -> tuple[torch.Tensor]:
        return (torch.zeros((1, batch_size, 1), device=device),)

    def forward_eval(
        self,
        observations: torch.Tensor,
        state: tuple[torch.Tensor],
    ) -> tuple[SimpleNamespace, torch.Tensor, tuple[torch.Tensor]]:
        self.observations.append(observations.clone())
        return (
            SimpleNamespace(mean=torch.zeros((1, 2))),
            torch.zeros((1, 1)),
            state,
        )


def test_scrambler_uses_same_agent_past_order_and_preserves_current_state() -> None:
    scrambler = DoorTemporalOrderScrambler(
        agent_count=2,
        seed=TEMPORAL_ORDER_ABLATION_SEED,
    )
    for step in range(4):
        current = _observations(step, 100 + step)
        assert torch.equal(scrambler.transform(current), current)

    current = _observations(4, 104)
    scrambled = scrambler.transform(current)

    assert scrambler.past_lag_permutation == (3, 1, 4, 2)
    assert torch.all(scrambled[0, :CAMERA_STOP] == 1.0)
    assert torch.all(scrambled[1, :CAMERA_STOP] == 101.0)
    assert torch.equal(
        scrambled[:, CAMERA_STOP:SENSOR_STOP],
        current[:, CAMERA_STOP:SENSOR_STOP],
    )
    assert torch.all(
        scrambled[0, SENSOR_STOP:DOOR_POLICY_OBS_DIM] == 2_001.0
    )
    assert torch.equal(
        scrambled[:, DOOR_POLICY_OBS_DIM:],
        current[:, DOOR_POLICY_OBS_DIM:],
    )


def test_scrambler_clears_only_terminal_agent_history() -> None:
    scrambler = DoorTemporalOrderScrambler(
        agent_count=2,
        seed=TEMPORAL_ORDER_ABLATION_SEED,
    )
    for step in range(5):
        scrambler.transform(_observations(step, 100 + step))
    scrambler.clear(torch.tensor((1.0, 0.0)))

    current = _observations(5, 105)
    scrambled = scrambler.transform(current)

    assert torch.all(scrambled[0, :CAMERA_STOP] == 5.0)
    assert torch.all(scrambled[1, :CAMERA_STOP] == 104.0)
    assert torch.equal(
        scrambled[:, CAMERA_STOP:SENSOR_STOP],
        current[:, CAMERA_STOP:SENSOR_STOP],
    )


def test_temporal_report_records_causal_limitations_and_mission_deltas() -> None:
    scrambler = DoorTemporalOrderScrambler(
        agent_count=2,
        seed=TEMPORAL_ORDER_ABLATION_SEED,
    )
    report = build_temporal_order_ablation(
        {"success_rate": 0.8, "outside_fov_success_rate": 0.7, "collision_rate": 0.01},
        {"success_rate": 0.6, "outside_fov_success_rate": 0.4, "collision_rate": 0.03},
        scrambler,
    )

    assert report["label"] == "causal_visual_temporal_order_scramble"
    assert report["mechanism"]["seed"] == TEMPORAL_ORDER_ABLATION_SEED
    assert report["mechanism"]["past_lag_permutation"] == [3, 1, 4, 2]
    assert report["mechanism"]["same_agent_only"] is True
    assert report["mechanism"]["terminal_cleared"] is True
    assert report["delta_vs_carried_ordered"]["success_rate"] == pytest.approx(
        -0.2
    )
    assert report["delta_vs_carried_ordered"]["collision_rate"] == pytest.approx(
        0.02
    )
    assert report["causal_online_limitations"]


def test_promotion_evaluator_applies_temporal_order_only_to_policy_input() -> None:
    policy = _RecordingPolicy()
    puffer = _TemporalPuffer()

    result = evaluate_promotion_door_policy(
        policy,
        {"env": {}, "vec": {}},
        puffer,
        steps=6,
        seed=31,
        camera_mask=False,
        temporal_order_seed=TEMPORAL_ORDER_ABLATION_SEED,
    )

    assert result["condition"]["temporal_order"] == "scrambled"
    assert torch.all(policy.observations[4][:, :CAMERA_STOP] == 1.0)
    assert torch.all(
        policy.observations[4][:, CAMERA_STOP:SENSOR_STOP] == 1_004.0
    )
    assert torch.all(
        policy.observations[4][:, SENSOR_STOP:DOOR_POLICY_OBS_DIM] == 2_001.0
    )
    assert torch.all(puffer.vec.observations[:, :CAMERA_STOP] == 6.0)


def test_promotion_evaluator_applies_camera_delay_only_to_policy_input() -> None:
    policy = _RecordingPolicy()
    puffer = _TemporalPuffer()
    transform = DoorCameraLatencyTransform(
        agent_count=1,
        delay_steps=1,
        control_dt_s=0.1,
        maximum_evidence_age_s=1.0,
    )

    result = evaluate_promotion_door_policy(
        policy,
        {"env": {}, "vec": {}},
        puffer,
        steps=3,
        seed=31,
        camera_mask=False,
        observation_transform=transform,
    )

    assert result["condition"]["observation_challenge"] is True
    assert torch.all(policy.observations[2][:, :CAMERA_STOP] == 1.0)
    assert torch.all(
        policy.observations[2][:, CAMERA_STOP:SENSOR_STOP] == 1_002.0
    )
    assert torch.all(puffer.vec.observations[:, :CAMERA_STOP] == 3.0)


def test_promotion_evaluator_rejects_combined_observation_challenges() -> None:
    transform = DoorCameraLatencyTransform(
        agent_count=1,
        delay_steps=1,
        control_dt_s=0.1,
        maximum_evidence_age_s=1.0,
    )

    with pytest.raises(ValueError, match="run in isolation"):
        evaluate_promotion_door_policy(
            _RecordingPolicy(),
            {"env": {}, "vec": {}},
            _TemporalPuffer(),
            steps=1,
            seed=31,
            camera_mask=True,
            observation_transform=transform,
        )
