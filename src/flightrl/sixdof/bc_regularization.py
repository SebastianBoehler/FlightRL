from __future__ import annotations

import torch

from .replay_loss import weighted_envelope_loss


VELOCITY_SLICE = slice(3, 6)
QUATERNION_SLICE = slice(6, 10)
HORIZONTAL_RANGE_SLICE = slice(18, 22)


def bc_regularization_loss(
    prediction: torch.Tensor,
    observations: torch.Tensor,
    *,
    envelope_coef: float = 0.0,
    action_abs_limit: float = 0.85,
    open_space_neutral_coef: float = 0.0,
    open_drift_brake_coef: float = 0.0,
    open_space_clearance_m: float = 0.85,
    neutral_speed_m_s: float = 0.20,
    drift_speed_m_s: float = 0.45,
    drift_brake_gain: float = 0.35,
    room_max_range_m: float = 4.0,
) -> torch.Tensor:
    loss = prediction.new_tensor(0.0)
    if envelope_coef > 0.0:
        loss = loss + envelope_coef * weighted_envelope_loss(prediction, action_abs_limit, None)
    if open_space_neutral_coef > 0.0:
        loss = loss + open_space_neutral_coef * open_space_neutral_loss(
            prediction,
            observations,
            open_space_clearance_m=open_space_clearance_m,
            neutral_speed_m_s=neutral_speed_m_s,
            room_max_range_m=room_max_range_m,
        )
    if open_drift_brake_coef > 0.0:
        loss = loss + open_drift_brake_coef * open_drift_brake_loss(
            prediction,
            observations,
            open_space_clearance_m=open_space_clearance_m,
            drift_speed_m_s=drift_speed_m_s,
            drift_brake_gain=drift_brake_gain,
            room_max_range_m=room_max_range_m,
        )
    return loss


def open_space_neutral_loss(
    prediction: torch.Tensor,
    observations: torch.Tensor,
    *,
    open_space_clearance_m: float = 0.85,
    neutral_speed_m_s: float = 0.20,
    room_max_range_m: float = 4.0,
) -> torch.Tensor:
    velocity = observations[:, VELOCITY_SLICE] * 3.0
    horizontal_speed = torch.linalg.vector_norm(velocity[:, :2], dim=1)
    horizontal_clearance = torch.amin(observations[:, HORIZONTAL_RANGE_SLICE] * room_max_range_m, dim=1)
    open_low_speed = (horizontal_clearance >= open_space_clearance_m) & (horizontal_speed <= neutral_speed_m_s)
    if not bool(torch.any(open_low_speed)):
        return prediction.new_tensor(0.0)
    return torch.mean(prediction[open_low_speed].pow(2))


def open_drift_brake_loss(
    prediction: torch.Tensor,
    observations: torch.Tensor,
    *,
    open_space_clearance_m: float = 0.85,
    drift_speed_m_s: float = 0.45,
    drift_brake_gain: float = 0.35,
    room_max_range_m: float = 4.0,
) -> torch.Tensor:
    velocity = observations[:, VELOCITY_SLICE] * 3.0
    body_velocity = torch.einsum("nij,ni->nj", quat_to_matrix(observations[:, QUATERNION_SLICE]), velocity)
    horizontal = body_velocity[:, :2]
    speed = torch.linalg.vector_norm(horizontal, dim=1)
    horizontal_clearance = torch.amin(observations[:, HORIZONTAL_RANGE_SLICE] * room_max_range_m, dim=1)
    active = (horizontal_clearance >= open_space_clearance_m) & (speed >= drift_speed_m_s)
    if not bool(torch.any(active)):
        return prediction.new_tensor(0.0)
    brake_direction = -horizontal / torch.clamp(speed[:, None], min=0.05)
    horizontal_control = torch.stack([prediction[:, 2], -prediction[:, 1]], dim=1)
    alignment = torch.sum(horizontal_control * brake_direction, dim=1)
    wrong_way = torch.relu(-alignment)
    weak_brake = torch.relu(torch.minimum(speed, speed.new_tensor(1.8)) * drift_brake_gain - alignment)
    return torch.mean(wrong_way[active].pow(2) + weak_brake[active].pow(2))


def quat_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    q = quaternion / torch.clamp(torch.linalg.vector_norm(quaternion, dim=1, keepdim=True), min=1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    matrix = torch.empty((q.shape[0], 3, 3), dtype=q.dtype, device=q.device)
    matrix[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[:, 0, 1] = 2.0 * (x * y - z * w)
    matrix[:, 0, 2] = 2.0 * (x * z + y * w)
    matrix[:, 1, 0] = 2.0 * (x * y + z * w)
    matrix[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[:, 1, 2] = 2.0 * (y * z - x * w)
    matrix[:, 2, 0] = 2.0 * (x * z - y * w)
    matrix[:, 2, 1] = 2.0 * (y * z + x * w)
    matrix[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix
