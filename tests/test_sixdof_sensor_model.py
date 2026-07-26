from __future__ import annotations

from pathlib import Path

import numpy as np

from flightrl.sim2real.live_profile import build_live_sim_profile, range_dropout_probability
from flightrl.sixdof import SixDofCrazyflieEnv
from flightrl.sixdof.sensor_model import SixDofSensorProfile, resolve_sensor_profile


HEADER = (
    "host_time_s,sys.isFlying,sys.isTumbled,stateEstimate.roll,stateEstimate.pitch,"
    "stateEstimate.x,stateEstimate.y,stateEstimate.z,stateEstimate.vx,stateEstimate.vy,stateEstimate.vz,"
    "gyro.x,gyro.y,gyro.z,range.front,range.back,range.left,range.right,range.up,range.zrange,"
    "min_horizontal_range_m,pm.vbat\n"
)


def test_live_profile_builder_extracts_sensor_knobs(tmp_path: Path) -> None:
    flight = tmp_path / "flight.csv"
    stationary = tmp_path / "stationary.csv"
    flight.write_text(
        HEADER
        + "\n".join(
            f"{i * 0.01},1,0,1,2,0,0,0.5,0.02,0.01,0,2,3,4,{200+i},{800+i},500,600,1200,500,0.20,3.9"
            for i in range(12)
        )
        + "\n"
    )
    stationary.write_text(
        HEADER
        + "\n".join(
            f"{i * 0.01},0,0,0,0,{0.001 * (i % 2)},0,0.02,0.30,0,0,{i % 2},0,0,{500 + (i % 2) * 20},700,800,900,1200,20,0.5,4.0"
            for i in range(20)
        )
        + "\n"
    )

    report = build_live_sim_profile(flight_logs=[flight], stationary_logs=[stationary], name="unit")

    assert report["summary"]["flight_rows"] == 12
    assert report["summary"]["stationary_rows"] == 20
    assert report["sensor_profile"]["range_noise_std_m"] > 0.0
    assert report["sensor_profile"]["velocity_noise_std_m_s"] < 0.01
    assert report["sensor_profile"]["action_lag_s"] > 0.0


def test_range_dropout_ignores_open_space_no_return_values() -> None:
    values = [32766, 32766, 500, 32766, 520, 32766, 32766, 700, 705, 710, 32766]
    rows = [{"range.front": str(value)} for value in values]

    assert range_dropout_probability(rows) == 0.5


def test_sensor_profile_adds_observation_dropout_and_action_lag() -> None:
    profile = SixDofSensorProfile(range_dropout_prob=1.0, action_lag_s=0.03)
    env = SixDofCrazyflieEnv(num_envs=4, seed=3, task="obstacle_avoidance", sensor_profile=profile)

    obs, _ = env.reset(seed=3)
    assert np.allclose(obs[:, 18:24], 1.0)

    action = np.ones((4, 4), dtype=np.float32)
    env.step(action)
    assert np.all(env.previous_action < 1.0)
    assert np.all(env.previous_action > 0.0)
    assert np.any(env.ranges_m[:, :4] < env.room.max_range_m)


def test_deckless_sensor_profile_masks_range_observation_without_disabling_room_model() -> None:
    profile = resolve_sensor_profile("deckless")
    env = SixDofCrazyflieEnv(num_envs=4, seed=13, task="position_yaw", sensor_profile=profile)

    obs, _ = env.reset(seed=13)

    assert profile.range_observation_enabled is False
    assert np.allclose(obs[:, 18:24], 1.0)
    assert np.any(env.ranges_m[:, :4] < env.room.max_range_m)


def test_sensor_profile_json_can_disable_range_observation(tmp_path: Path) -> None:
    path = tmp_path / "sensor.json"
    path.write_text('{"sensor_profile": {"name": "unit", "range_observation_enabled": false}}')

    profile = resolve_sensor_profile(path)

    assert profile.name == "unit"
    assert profile.range_observation_enabled is False
