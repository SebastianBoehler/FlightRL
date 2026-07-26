from __future__ import annotations

import json
from pathlib import Path

from flightrl.sixdof.physics import resolve_physics_profile
from flightrl.sixdof.puffer_calibration import PhysicsSweepGrid, candidate_profiles, profile_score


def test_candidate_profiles_use_base_rates_and_grid_values() -> None:
    profiles = candidate_profiles(
        "crazyflie_brushless",
        PhysicsSweepGrid(linear_drag=(0.04, 0.08), rate_tau_s=(0.03,), motor_tau_s=(0.02,), thrust_scale=(0.70, 0.80)),
    )

    assert len(profiles) == 4
    assert {profile.linear_drag for profile in profiles} == {0.04, 0.08}
    assert {profile.thrust_scale for profile in profiles} == {0.70, 0.80}
    assert all(profile.max_rate_rad_s == (6.0, 6.0, 4.0) for profile in profiles)


def test_profile_score_prefers_target_like_metrics() -> None:
    target = {
        "open_space_horizontal_speed_p95_m_s": 0.8,
        "horizontal_speed_p95_m_s": 0.82,
        "mean_position_error_m": 0.15,
        "clearance_p01_m": 0.14,
        "tilt_p95_deg": 7.0,
    }
    close = {**target, "open_space_horizontal_speed_p95_m_s": 0.79}
    far = {**target, "open_space_horizontal_speed_p95_m_s": 0.50, "mean_position_error_m": 0.30}

    assert profile_score(target, close) < profile_score(target, far)


def test_resolve_physics_profile_loads_json(tmp_path: Path) -> None:
    path = tmp_path / "physics.json"
    path.write_text(
        json.dumps(
            {
                "physics_profile": {
                    "mass_kg": 0.036,
                    "gravity_m_s2": 9.81,
                    "linear_drag": 0.07,
                    "rate_tau_s": 0.05,
                    "thrust_scale": 0.76,
                    "max_rate_rad_s": [5.5, 5.5, 3.5],
                    "motor_tau_s": 0.04,
                }
            }
        )
    )

    profile = resolve_physics_profile(str(path))
    assert profile.linear_drag == 0.07
    assert profile.max_rate_rad_s == (5.5, 5.5, 3.5)
