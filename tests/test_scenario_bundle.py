from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flightrl.navigation.mission_spec import (
    MissionCommand,
    ResolvedMissionPlan,
    ResolvedMissionStep,
    TargetAnchor,
)
from flightrl.scenario_bundle import (
    SCENARIO_BUNDLE_SCHEMA,
    compile_scenario_bundle,
    load_scenario_bundle,
    write_scenario_bundle,
)
from flightrl.sixdof.geometry import AxisAlignedObstacle, BoxRoom
from flightrl.sixdof.physics import SixDofPhysicsProfile
from flightrl.sixdof.sensor_model import SixDofSensorProfile


def _mission() -> ResolvedMissionPlan:
    return ResolvedMissionPlan(
        source_text="go to the gate, then hold",
        steps=(
            ResolvedMissionStep(
                command=MissionCommand.GO_TO,
                target_index=2,
                anchor=TargetAnchor.APPROACH,
                target_xyz_m=(1.0, -0.5, 0.8),
                target_yaw_rad=0.25,
                duration_s=0.0,
                speed_scale=0.75,
            ),
            ResolvedMissionStep(
                command=MissionCommand.HOLD,
                target_index=2,
                anchor=TargetAnchor.APPROACH,
                target_xyz_m=(1.0, -0.5, 0.8),
                target_yaw_rad=0.25,
                duration_s=2.0,
                speed_scale=1.0,
            ),
        ),
    )


def _bundle():
    return compile_scenario_bundle(
        vehicle=SixDofPhysicsProfile(
            mass_kg=0.25,
            gravity_m_s2=9.81,
            linear_drag=0.12,
            rate_tau_s=0.04,
            thrust_scale=1.1,
            max_rate_rad_s=(7.0, 8.0, 5.0),
            motor_tau_s=0.03,
        ),
        terrain=BoxRoom(
            x_min=-2.0,
            x_max=3.0,
            y_min=-1.5,
            y_max=2.5,
            z_min=0.0,
            z_max=2.0,
            max_range_m=4.0,
            obstacles=(
                AxisAlignedObstacle(
                    x_min=0.2,
                    x_max=0.6,
                    y_min=-0.3,
                    y_max=0.3,
                    z_min=0.0,
                    z_max=1.2,
                ),
            ),
        ),
        sensor=SixDofSensorProfile(
            name="demo",
            range_observation_enabled=True,
            state_noise_std_m=0.01,
            velocity_noise_std_m_s=0.02,
            body_rate_noise_std_rad_s=0.03,
            range_noise_std_m=0.04,
            range_dropout_prob=0.05,
            action_lag_s=0.06,
        ),
        mission=_mission(),
    )


def test_compile_scenario_bundle_freezes_explicit_runtime_contract() -> None:
    bundle = _bundle()

    assert bundle.manifest["schema"] == SCENARIO_BUNDLE_SCHEMA
    assert bundle.manifest["authority"] == "simulation_only"
    assert bundle.manifest["deployment_authority"] is False
    assert bundle.manifest["frames"] == {
        "world": "local_cartesian_right_handed_z_up",
        "body": "front_left_up",
        "quaternion": "world_from_body_wxyz",
    }
    np.testing.assert_array_equal(
        bundle.arrays["vehicle_physics"],
        np.asarray(
            [0.25, 9.81, 0.12, 0.04, 1.1, 7.0, 8.0, 5.0, 0.03],
            dtype="<f4",
        ),
    )
    np.testing.assert_array_equal(
        bundle.arrays["terrain_obstacles"],
        np.asarray([[0.2, 0.6, -0.3, 0.3, 0.0, 1.2]], dtype="<f4"),
    )
    assert bundle.arrays["mission_steps"].shape == (2, 9)
    with pytest.raises(ValueError, match="read-only"):
        bundle.arrays["vehicle_physics"][0] = 1.0


def test_scenario_bundle_is_deterministic_and_detects_disk_tampering(
    tmp_path: Path,
) -> None:
    first = _bundle()
    second = _bundle()
    assert first.manifest == second.manifest

    output = write_scenario_bundle(first, tmp_path / "scenario")
    loaded = load_scenario_bundle(output)
    assert loaded.manifest == first.manifest
    for name in first.arrays:
        np.testing.assert_array_equal(loaded.arrays[name], first.arrays[name])

    vehicle_path = output / "vehicle_physics.npy"
    values = np.load(vehicle_path, allow_pickle=False)
    values[0] = 0.5
    np.save(vehicle_path, values, allow_pickle=False)

    with pytest.raises(ValueError, match="SHA-256"):
        load_scenario_bundle(output)


def test_scenario_bundle_rejects_mission_target_outside_terrain() -> None:
    mission = ResolvedMissionPlan(
        source_text="go outside",
        steps=(
            ResolvedMissionStep(
                command=MissionCommand.GO_TO,
                target_index=0,
                anchor=TargetAnchor.CENTER,
                target_xyz_m=(5.0, 0.0, 0.8),
                target_yaw_rad=0.0,
                duration_s=0.0,
                speed_scale=1.0,
            ),
        ),
    )

    with pytest.raises(ValueError, match="outside navigable terrain"):
        compile_scenario_bundle(
            vehicle=SixDofPhysicsProfile(),
            terrain=BoxRoom(),
            sensor=SixDofSensorProfile(),
            mission=mission,
        )


def test_scenario_bundle_rejects_nonfinite_resolved_mission_values() -> None:
    mission = ResolvedMissionPlan(
        source_text="invalid yaw",
        steps=(
            ResolvedMissionStep(
                command=MissionCommand.HOLD,
                target_index=-1,
                anchor=TargetAnchor.PREFERRED,
                target_xyz_m=(0.0, 0.0, 0.8),
                target_yaw_rad=float("nan"),
                duration_s=1.0,
                speed_scale=1.0,
            ),
        ),
    )

    with pytest.raises(ValueError, match="finite float32"):
        compile_scenario_bundle(
            vehicle=SixDofPhysicsProfile(),
            terrain=BoxRoom(),
            sensor=SixDofSensorProfile(),
            mission=mission,
        )


def test_empty_mission_compiles_for_diagnostic_scenario(tmp_path: Path) -> None:
    bundle = compile_scenario_bundle(
        vehicle=SixDofPhysicsProfile(), terrain=BoxRoom(), sensor=SixDofSensorProfile(),
        mission=ResolvedMissionPlan(source_text="diagnostic", steps=()),
    )
    loaded = load_scenario_bundle(write_scenario_bundle(bundle, tmp_path / "scenario"))
    assert loaded.arrays["mission_steps"].shape == (0, 9)
