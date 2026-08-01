from __future__ import annotations

import torch

from flightrl.puffer4_door_challenges import (
    DoorCameraLatencyTransform,
    DoorPixelNoiseTransform,
)
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


def _observation(
    current: float,
    *,
    sensor: float,
    phase: int = 1,
    confidence: float = 0.8,
    age: float = 0.1,
    agents: int = 2,
) -> torch.Tensor:
    result = torch.zeros((agents, DOOR_OBS_DIM), dtype=torch.float32)
    result[:, :DOOR_PIXELS] = current
    result[:, CAMERA_STOP:SENSOR_STOP] = sensor
    result[:, SENSOR_STOP + phase] = 1.0
    result[:, PHASE_STOP] = confidence
    result[:, PHASE_STOP + 1 : PHASE_STOP + 4] = 0.2
    result[:, PHASE_STOP + 4] = age
    result[:, DOOR_POLICY_OBS_DIM:] = sensor + 1_000.0
    return result


def test_noise_is_deterministic_and_preserves_nonvisual_contract() -> None:
    observations = _observation(68.0 / 255.0, sensor=7.0)
    first = DoorPixelNoiseTransform(agent_count=2, seed=31)
    second = DoorPixelNoiseTransform(agent_count=2, seed=31)

    transformed = first.transform(observations)

    assert torch.equal(transformed, second.transform(observations))
    assert not torch.equal(
        transformed[:, :DOOR_PIXELS],
        observations[:, :DOOR_PIXELS],
    )
    assert torch.equal(
        transformed[:, CAMERA_STOP:],
        observations[:, CAMERA_STOP:],
    )
    mask = torch.as_tensor(door_self_mask(48, 64).reshape(-1).copy())
    assert torch.equal(
        transformed[:, :DOOR_PIXELS][:, mask],
        observations[:, :DOOR_PIXELS][:, mask],
    )


def test_noise_recomputes_signed_delta_and_motion() -> None:
    transform = DoorPixelNoiseTransform(agent_count=2, seed=47)
    first = transform.transform(_observation(68.0 / 255.0, sensor=1.0))
    second = transform.transform(_observation(102.0 / 255.0, sensor=2.0))

    assert torch.count_nonzero(first[:, DOOR_PIXELS:CAMERA_STOP]) == 0
    expected_delta = (
        second[:, :DOOR_PIXELS] - first[:, :DOOR_PIXELS]
    )
    torch.testing.assert_close(
        second[:, DOOR_PIXELS : 2 * DOOR_PIXELS],
        expected_delta,
    )
    assert torch.equal(
        second[:, 2 * DOOR_PIXELS:CAMERA_STOP],
        (expected_delta.abs() >= 0.08).to(dtype=torch.float32),
    )


def test_noise_terminal_clear_resets_only_one_agent_history() -> None:
    transform = DoorPixelNoiseTransform(agent_count=2, seed=53)
    transform.transform(_observation(68.0 / 255.0, sensor=1.0))
    transform.clear(torch.tensor((1.0, 0.0)))
    result = transform.transform(_observation(102.0 / 255.0, sensor=2.0))

    assert torch.count_nonzero(
        result[0, DOOR_PIXELS:CAMERA_STOP]
    ) == 0
    assert torch.count_nonzero(
        result[1, DOOR_PIXELS : 2 * DOOR_PIXELS]
    ) > 0


def test_latency_delays_camera_bundle_but_keeps_current_sensors() -> None:
    transform = DoorCameraLatencyTransform(
        agent_count=2,
        delay_steps=2,
        control_dt_s=1.0 / 10.0,
        maximum_evidence_age_s=1.0,
    )
    warmup = transform.transform(
        _observation(0.1, sensor=10.0, phase=1, age=0.2)
    )
    transform.transform(_observation(0.2, sensor=20.0, phase=2, age=0.3))
    delayed = transform.transform(
        _observation(0.3, sensor=30.0, phase=2, age=0.4)
    )

    assert torch.count_nonzero(warmup[:, :CAMERA_STOP]) == 0
    assert torch.equal(
        warmup[:, SENSOR_STOP:PHASE_STOP],
        torch.tensor(((1.0, 0.0, 0.0, 0.0),) * 2),
    )
    assert torch.all(warmup[:, PHASE_STOP + 4] == 1.0)
    assert torch.all(delayed[:, :DOOR_PIXELS] == 0.1)
    assert torch.all(delayed[:, CAMERA_STOP:SENSOR_STOP] == 30.0)
    assert torch.equal(
        delayed[:, SENSOR_STOP:PHASE_STOP],
        torch.tensor(((0.0, 1.0, 0.0, 0.0),) * 2),
    )
    assert torch.allclose(
        delayed[:, PHASE_STOP + 4],
        torch.full((2,), 0.4),
    )
    assert torch.all(delayed[:, DOOR_POLICY_OBS_DIM:] == 1_030.0)


def test_latency_expires_stale_evidence_into_recovery() -> None:
    transform = DoorCameraLatencyTransform(
        agent_count=2,
        delay_steps=2,
        control_dt_s=0.2,
        maximum_evidence_age_s=0.5,
    )
    historical = _observation(
        0.1,
        sensor=1.0,
        phase=2,
        confidence=0.9,
        age=0.4,
    )
    transform.transform(historical)
    transform.transform(_observation(0.2, sensor=2.0))
    delayed = transform.transform(_observation(0.3, sensor=3.0))

    assert torch.equal(
        delayed[:, SENSOR_STOP:PHASE_STOP],
        torch.tensor(((0.0, 0.0, 0.0, 1.0),) * 2),
    )
    assert torch.count_nonzero(delayed[:, PHASE_STOP : PHASE_STOP + 4]) == 0
    assert torch.all(delayed[:, PHASE_STOP + 4] == 1.0)


def test_latency_terminal_clear_prevents_cross_episode_history() -> None:
    transform = DoorCameraLatencyTransform(
        agent_count=2,
        delay_steps=1,
        control_dt_s=0.1,
        maximum_evidence_age_s=1.0,
    )
    transform.transform(_observation(0.1, sensor=1.0))
    transform.clear(torch.tensor((1.0, 0.0)))
    result = transform.transform(_observation(0.2, sensor=2.0))

    assert torch.count_nonzero(result[0, :CAMERA_STOP]) == 0
    assert torch.all(result[1, :DOOR_PIXELS] == 0.1)
    assert torch.all(result[:, CAMERA_STOP:SENSOR_STOP] == 2.0)


def test_challenge_reports_record_single_intervention_and_limitations() -> None:
    noise = DoorPixelNoiseTransform(agent_count=2, seed=11)
    latency = DoorCameraLatencyTransform(
        agent_count=2,
        delay_steps=6,
        control_dt_s=1.0 / 65.0,
        maximum_evidence_age_s=1.0,
    )

    assert noise.mechanism_report()["single_intervention"] == "pixel_noise"
    assert noise.mechanism_report()["detector_evidence_recomputed"] is False
    assert latency.mechanism_report()["delay_ms"] == 6 / 65 * 1_000
    assert latency.mechanism_report()["camera_bundle_delayed_together"] is True
