from __future__ import annotations

import torch

from flightrl.sixdof.bc_regularization import bc_regularization_loss, open_drift_brake_loss, open_space_neutral_loss


def observation(*, speed_m_s: float, clearance_m: float) -> torch.Tensor:
    obs = torch.zeros(1, 28)
    obs[:, 3] = speed_m_s / 3.0
    obs[:, 6] = 1.0
    obs[:, 18:22] = clearance_m / 4.0
    return obs


def test_open_space_neutral_loss_penalizes_low_speed_open_bias() -> None:
    biased = torch.full((1, 4), 0.2)
    neutral = torch.zeros((1, 4))
    obs = observation(speed_m_s=0.05, clearance_m=1.2)

    assert open_space_neutral_loss(biased, obs) > open_space_neutral_loss(neutral, obs)


def test_open_space_neutral_loss_ignores_drift_recovery_samples() -> None:
    biased = torch.full((1, 4), 0.2)
    obs = observation(speed_m_s=0.8, clearance_m=1.2)

    assert open_space_neutral_loss(biased, obs) == 0.0


def test_bc_regularization_combines_envelope_and_open_space_neutral_terms() -> None:
    prediction = torch.full((1, 4), 0.4)
    obs = observation(speed_m_s=0.05, clearance_m=1.2)

    loss = bc_regularization_loss(
        prediction,
        obs,
        envelope_coef=1.0,
        action_abs_limit=0.1,
        open_space_neutral_coef=1.0,
    )

    assert loss > open_space_neutral_loss(prediction, obs)


def test_open_drift_brake_loss_prefers_body_frame_braking_action() -> None:
    obs = observation(speed_m_s=1.0, clearance_m=1.2)
    wrong = torch.tensor([[0.0, 0.0, 0.3, 0.0]])
    brake = torch.tensor([[0.0, 0.0, -0.5, 0.0]])

    assert open_drift_brake_loss(wrong, obs) > open_drift_brake_loss(brake, obs)


def test_open_drift_brake_loss_ignores_low_speed_open_samples() -> None:
    obs = observation(speed_m_s=0.05, clearance_m=1.2)
    wrong = torch.tensor([[0.0, 0.0, 0.3, 0.0]])

    assert open_drift_brake_loss(wrong, obs) == 0.0
