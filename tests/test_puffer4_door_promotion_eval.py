from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest
import torch

from flightrl.puffer4_door_promotion_eval import (
    build_recurrence_reset_ablation,
    episode_evidence,
    evaluate_promotion_door_policy,
    wilson_interval,
)


class _FakeVec:
    total_agents = 2
    obs_size = 4

    def __init__(self) -> None:
        self.observations = torch.zeros((self.total_agents, self.obs_size))
        self.terminals = torch.zeros(self.total_agents)
        self.obs_ptr = self.observations.data_ptr()
        self.terminals_ptr = self.terminals.data_ptr()
        self.steps = 0
        self.actions: list[torch.Tensor] = []

    def reset(self) -> None:
        self.observations.zero_()
        self.terminals.zero_()

    def cpu_step(self, actions_ptr: int) -> None:
        raw = (ctypes.c_float * (self.total_agents * 2)).from_address(
            actions_ptr
        )
        self.actions.append(
            torch.frombuffer(raw, dtype=torch.float32)
            .reshape(self.total_agents, 2)
            .clone()
        )
        self.steps += 1
        self.observations.add_(1.0)
        self.terminals.zero_()

    def log(self) -> dict[str, float]:
        return {
            "n": 10.0,
            "success_rate": 0.6,
            "collision_rate": 0.1,
            "outside_fov_episode_fraction": 0.5,
            "outside_fov_success_fraction": 0.3,
        }

    def close(self) -> None:
        return None


class _FakePuffer:
    def __init__(self) -> None:
        self.vecs: list[_FakeVec] = []
        self._C = SimpleNamespace(create_vec=self._create_vec, gpu=0)

    def _create_vec(self, _args: dict, _gpu: int) -> _FakeVec:
        vec = _FakeVec()
        self.vecs.append(vec)
        return vec

    def _cpu_tensor(
        self,
        pointer: int,
        _shape: tuple[int, ...],
        _dtype: torch.dtype,
    ) -> torch.Tensor:
        vec = self.vecs[-1]
        return {
            vec.obs_ptr: vec.observations,
            vec.terminals_ptr: vec.terminals,
        }[pointer]


class _StatePolicy:
    def __init__(self, yaw: float = 0.8, fail_at: int | None = None) -> None:
        self.yaw = yaw
        self.fail_at = fail_at
        self.calls = 0
        self.seen_states: list[torch.Tensor] = []

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
        self.calls += 1
        self.seen_states.append(state[0].clone())
        yaw = self.yaw
        if self.calls == self.fail_at:
            yaw = float("inf")
        means = torch.full((observations.shape[0], 2), 0.5)
        means[:, 1] = yaw
        values = torch.zeros((observations.shape[0], 1))
        return SimpleNamespace(mean=means), values, (state[0] + 1.0,)


def _args() -> dict:
    return {"env": {"seed": 1, "camera_mask": 0.0}, "vec": {}}


def test_episode_evidence_recovers_counts_and_wilson_intervals() -> None:
    evidence = episode_evidence(
        {
            "n": 100.0,
            "success_rate": 0.8,
            "collision_rate": 0.03,
            "outside_fov_episode_fraction": 0.4,
            "outside_fov_success_fraction": 0.3,
        }
    )

    assert evidence["counts"] == {
        "episodes": 100,
        "successes": 80,
        "collisions": 3,
        "outside_fov_episodes": 40,
        "outside_fov_successes": 30,
    }
    assert evidence["wilson_95"]["success_rate"]["low"] < 0.8
    assert evidence["wilson_95"]["success_rate"]["high"] > 0.8
    assert evidence["wilson_95"]["outside_fov_success_rate"]["estimate"] == 0.75


def test_wilson_interval_handles_empty_denominator() -> None:
    assert wilson_interval(0, 0) is None
    with pytest.raises(ValueError, match="cannot exceed"):
        wilson_interval(2, 1)


def test_episode_evidence_rejects_non_integral_native_totals() -> None:
    with pytest.raises(ValueError, match="integer count"):
        episode_evidence(
            {
                "n": 3.0,
                "success_rate": 0.2,
                "collision_rate": 0.0,
                "outside_fov_episode_fraction": 0.0,
                "outside_fov_success_fraction": 0.0,
            }
        )


def test_evaluator_reports_performance_finiteness_and_yaw_cap() -> None:
    puffer = _FakePuffer()
    policy = _StatePolicy(yaw=0.8)

    result = evaluate_promotion_door_policy(
        policy,
        _args(),
        puffer,
        steps=20,
        seed=19,
        camera_mask=False,
        agents=2,
        yaw_abs_limit_normalized=0.1,
    )

    assert result["status"] == "complete"
    assert result["finite_outputs"]["passed"] is True
    assert result["episode_evidence"]["counts"]["episodes"] == 10
    assert result["performance"]["agent_steps"] == 40
    assert result["performance"]["latency_warmup"]["excluded_batches"] == 16
    assert result["performance"]["latency_warmup"]["mission_steps_excluded"] == 0
    assert result["performance"]["closed_loop_batch_ms"]["p95"] >= 0.0
    assert result["yaw_cap"]["normalized_limit"] == 0.1
    assert result["yaw_cap"]["saturation_fraction"] == 1.0
    assert result["yaw_proposal_abs_p95"] == pytest.approx(0.8)
    assert result["yaw_action_p95"] == pytest.approx(0.1)
    assert all(
        torch.all(batch[:, 1].abs() <= 0.100001)
        for batch in puffer.vecs[0].actions
    )


def test_evaluator_reset_mode_never_carries_recurrent_state() -> None:
    carried_policy = _StatePolicy()
    evaluate_promotion_door_policy(
        carried_policy,
        _args(),
        _FakePuffer(),
        steps=3,
        seed=23,
        camera_mask=False,
        recurrent_mode="carried",
    )
    reset_policy = _StatePolicy()
    evaluate_promotion_door_policy(
        reset_policy,
        _args(),
        _FakePuffer(),
        steps=3,
        seed=23,
        camera_mask=False,
        recurrent_mode="reset_each_step",
    )

    assert carried_policy.seen_states[1].count_nonzero() > 0
    assert all(state.count_nonzero() == 0 for state in reset_policy.seen_states)


def test_evaluator_stops_before_stepping_a_non_finite_action() -> None:
    puffer = _FakePuffer()
    result = evaluate_promotion_door_policy(
        _StatePolicy(fail_at=2),
        _args(),
        puffer,
        steps=4,
        seed=29,
        camera_mask=False,
    )

    assert result["status"] == "aborted_non_finite"
    assert result["finite_outputs"]["policy_mean"] is False
    assert result["finite_outputs"]["passed"] is False
    assert puffer.vecs[0].steps == 1


def test_reset_ablation_is_not_mislabeled_as_temporal_shuffle() -> None:
    report = build_recurrence_reset_ablation(
        {"success_rate": 0.8, "outside_fov_success_rate": 0.7},
        {"success_rate": 0.5, "outside_fov_success_rate": 0.4},
    )

    assert report["label"] == "recurrent_state_reset_each_step"
    assert report["not_a_temporal_order_shuffle"] is True
    assert report["delta_vs_carried"]["success_rate"] == pytest.approx(-0.3)
