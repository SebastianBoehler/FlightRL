from __future__ import annotations

import csv
from pathlib import Path

from flightrl.sim2real.live_profile import (
    build_live_sim_profile,
    write_report as write_sensor_profile,
)
from flightrl.sim2real.noise import DEFAULT_COLUMNS
from flightrl.sixdof.signal_evidence import RANGE_SIGNALS, REPLAY_STATE_SIGNALS


def write_config(
    tmp_path: Path,
    name: str,
    *,
    measured: bool,
    include_noisy_state: bool = True,
) -> Path:
    path = tmp_path / name
    path.write_text(
        f"""
[environment]
dt = 0.02
action_mode = "motor_quad"

[sim2real]
measured = {str(measured).lower()}
source = "test_fixture"

[drone]
mass = 1.15
inertia = 0.09
arm_length = 0.23
drag = 0.14
angular_drag = 0.09
hover_thrust = 10.6
thrust_gain = 4.2
max_total_thrust = 19.5
max_pitch_torque = 2.2
actuator_tau = 0.11

[sensors]
include_noisy_state = {str(include_noisy_state).lower()}

[domain_randomization]
enabled = true
""".strip()
    )
    return path


def replay_signals(samples: int, state_rmse: float, range_rmse: float) -> dict:
    return {
        name: {
            "samples": samples,
            "rmse": range_rmse if name in RANGE_SIGNALS else state_rmse,
        }
        for name in (*REPLAY_STATE_SIGNALS, *RANGE_SIGNALS)
    }


def stationary_signals() -> dict:
    return {
        column: {"samples": 100, "valid_ratio": 1.0, "std": 0.01}
        for column in DEFAULT_COLUMNS
    }


def write_live_sensor_profile(tmp_path: Path) -> Path:
    flight = tmp_path / "profile_flight.csv"
    stationary = tmp_path / "profile_stationary.csv"
    fields = [
        "host_time_s",
        "sys.isFlying",
        "sys.isTumbled",
        "stateEstimate.roll",
        "stateEstimate.pitch",
        "stateEstimate.x",
        "stateEstimate.y",
        "stateEstimate.z",
        "stateEstimate.vx",
        "stateEstimate.vy",
        "stateEstimate.vz",
        "gyro.x",
        "gyro.y",
        "gyro.z",
        "range.front",
        "range.back",
        "range.left",
        "range.right",
        "range.up",
        "range.zrange",
    ]
    for path, flying in ((flight, 1), (stationary, 0)):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index in range(6):
                writer.writerow(
                    {
                        **{field: 0.001 * (index % 2) for field in fields},
                        "host_time_s": index * 0.02,
                        "sys.isFlying": flying,
                        "sys.isTumbled": 0,
                        **{
                            field: 500 + index % 2
                            for field in fields
                            if field.startswith("range.")
                        },
                    }
                )
    output = tmp_path / "sensor_profile.json"
    write_sensor_profile(
        build_live_sim_profile(
            flight_logs=[flight],
            stationary_logs=[stationary],
            name="measured_unit",
        ),
        output,
    )
    return output
