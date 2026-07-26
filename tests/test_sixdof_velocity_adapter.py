from __future__ import annotations

import numpy as np

from flightrl.hardware.avoidance_policy import AvoidanceCommand
from flightrl.hardware.sixdof_velocity_adapter import SixDofVelocityAdapterConfig, sixdof_action_to_velocity_command
from flightrl.sixdof import SixDofCrazyflieEnv
from flightrl.sixdof.env import euler_to_quat


def test_sixdof_action_adapter_clamps_horizontal_speed() -> None:
    env = SixDofCrazyflieEnv(num_envs=1, seed=1)
    command = sixdof_action_to_velocity_command(
        env,
        np.asarray([0.5, 1.0, 1.0, 1.0], dtype=np.float32),
        SixDofVelocityAdapterConfig(max_horizontal_speed_m_s=0.05, max_vertical_speed_m_s=0.04, max_yawrate_deg_s=10.0),
    )

    assert np.hypot(command.vx_m_s, command.vy_m_s) <= 0.0501
    assert abs(command.vz_m_s - 0.02) < 1e-6
    assert command.yawrate_deg_s == 10.0
    assert command.zdistance_m == env.target_position[0, 2]


def test_sixdof_action_adapter_rotates_body_intent_by_yaw() -> None:
    env = SixDofCrazyflieEnv(num_envs=1, seed=1)
    env.quaternion[0] = euler_to_quat(np.asarray([0.0]), np.asarray([0.0]), np.asarray([np.pi / 2]))[0]
    command = sixdof_action_to_velocity_command(
        env,
        np.asarray([0.0, 0.0, 0.5, 0.0], dtype=np.float32),
        SixDofVelocityAdapterConfig(max_horizontal_speed_m_s=1.0),
    )

    assert abs(command.vx_m_s) < 0.02
    assert command.vy_m_s > 0.1


def test_sixdof_action_adapter_can_blend_with_base_command() -> None:
    env = SixDofCrazyflieEnv(num_envs=1, seed=1)
    base = AvoidanceCommand(0.2, 0.0, 0.0, 0.5)
    command = sixdof_action_to_velocity_command(
        env,
        np.asarray([0.0, 0.0, -1.0, 0.0], dtype=np.float32),
        SixDofVelocityAdapterConfig(max_horizontal_speed_m_s=1.0, policy_blend=0.25),
        base=base,
    )

    assert command.vx_m_s < base.vx_m_s
    assert command.vz_m_s == 0.0
    assert command.zdistance_m == base.zdistance_m
