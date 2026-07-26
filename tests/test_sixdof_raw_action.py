from __future__ import annotations

import numpy as np
import pytest

from flightrl.hardware.sixdof_raw_action import RawPufferActionConfig, raw_action_to_manual_setpoint


def test_raw_action_maps_directly_to_manual_rate_setpoint() -> None:
    setpoint = raw_action_to_manual_setpoint(
        np.asarray([0.5, -0.25, 0.5, -1.0], dtype=np.float32),
        RawPufferActionConfig(
            hover_thrust_percent=48.0,
            thrust_scale=0.75,
            max_roll_rate_deg_s=300.0,
            max_pitch_rate_deg_s=400.0,
            max_yaw_rate_deg_s=200.0,
        ),
    )

    assert setpoint.thrust_percent == 66.0
    assert setpoint.roll_rate_deg_s == -75.0
    assert setpoint.pitch_rate_deg_s == 200.0
    assert setpoint.commander_pitch_rate_deg_s == -200.0
    assert setpoint.yaw_rate_deg_s == -200.0


def test_raw_action_raises_when_mapped_thrust_is_not_flyable_api_value() -> None:
    with pytest.raises(ValueError, match="thrust_percent"):
        raw_action_to_manual_setpoint(
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            RawPufferActionConfig(hover_thrust_percent=80.0, thrust_scale=0.75),
        )
