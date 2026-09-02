"""Compile typed FlightRL inputs into immutable simulation bundles."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from flightrl.artifact_identity import (
    bind_payload,
    require_bound_payload,
    sha256_bytes,
)
from flightrl.navigation.mission_spec import (
    MISSION_STEP_FIELDS,
    ResolvedMissionPlan,
)
from flightrl.sixdof.geometry import BoxRoom
from flightrl.sixdof.physics import SixDofPhysicsProfile
from flightrl.sixdof.sensor_model import SixDofSensorProfile


SCENARIO_BUNDLE_SCHEMA = "flightrl.scenario_bundle.v1"
FRAME_CONTRACT = {
    "world": "local_cartesian_right_handed_z_up",
    "body": "front_left_up",
    "quaternion": "world_from_body_wxyz",
}
VEHICLE_FIELDS = (
    "mass_kg",
    "gravity_m_s2",
    "linear_drag",
    "rate_tau_s",
    "thrust_scale",
    "max_rate_roll_rad_s",
    "max_rate_pitch_rad_s",
    "max_rate_yaw_rad_s",
    "motor_tau_s",
)
TERRAIN_BOUNDS_FIELDS = (
    "x_min_m",
    "x_max_m",
    "y_min_m",
    "y_max_m",
    "z_min_m",
    "z_max_m",
    "max_range_m",
)
TERRAIN_OBSTACLE_FIELDS = TERRAIN_BOUNDS_FIELDS[:6]
SENSOR_FIELDS = (
    "state_noise_std_m",
    "velocity_noise_std_m_s",
    "body_rate_noise_std_rad_s",
    "range_noise_std_m",
    "range_dropout_probability",
    "action_lag_s",
)
_ARRAY_FIELDS = {
    "vehicle_physics": VEHICLE_FIELDS,
    "terrain_bounds": TERRAIN_BOUNDS_FIELDS,
    "terrain_obstacles": TERRAIN_OBSTACLE_FIELDS,
    "sensor_parameters": SENSOR_FIELDS,
    "mission_steps": MISSION_STEP_FIELDS,
}


@dataclass(frozen=True, slots=True)
class CompiledScenarioBundle:
    manifest: Mapping[str, object]
    arrays: Mapping[str, np.ndarray]


def compile_scenario_bundle(
    *,
    vehicle: SixDofPhysicsProfile,
    terrain: BoxRoom,
    sensor: SixDofSensorProfile,
    mission: ResolvedMissionPlan,
) -> CompiledScenarioBundle:
    """Compile existing validated model types into one runtime contract."""
    _require_inputs(vehicle, terrain, sensor, mission)
    _require_mission_inside_terrain(mission, terrain)
    arrays = {
        "vehicle_physics": vehicle.as_array(),
        "terrain_bounds": np.asarray(
            [
                terrain.x_min,
                terrain.x_max,
                terrain.y_min,
                terrain.y_max,
                terrain.z_min,
                terrain.z_max,
                terrain.max_range_m,
            ],
            dtype="<f4",
        ),
        "terrain_obstacles": np.asarray(
            [
                [
                    obstacle.x_min,
                    obstacle.x_max,
                    obstacle.y_min,
                    obstacle.y_max,
                    obstacle.z_min,
                    obstacle.z_max,
                ]
                for obstacle in terrain.obstacles
            ],
            dtype="<f4",
        ).reshape(-1, 6),
        "sensor_parameters": np.asarray(
            [
                sensor.state_noise_std_m,
                sensor.velocity_noise_std_m_s,
                sensor.body_rate_noise_std_rad_s,
                sensor.range_noise_std_m,
                sensor.range_dropout_prob,
                sensor.action_lag_s,
            ],
            dtype="<f4",
        ),
        "mission_steps": np.asarray(mission.to_rows(), dtype="<f4").reshape(
            -1,
            len(MISSION_STEP_FIELDS),
        ),
    }
    frozen = {name: _freeze_array(value) for name, value in arrays.items()}
    manifest = bind_payload(
        {
            "schema": SCENARIO_BUNDLE_SCHEMA,
            "authority": "simulation_only",
            "deployment_authority": False,
            "frames": FRAME_CONTRACT,
            "sensor": {
                "name": sensor.name,
                "range_observation_enabled": sensor.range_observation_enabled,
            },
            "mission": {
                "contract_version": mission.contract_version,
                "source_text": mission.source_text,
            },
            "arrays": {
                name: _array_manifest(name, frozen[name], fields)
                for name, fields in _ARRAY_FIELDS.items()
            },
        }
    )
    return CompiledScenarioBundle(
        manifest=MappingProxyType(manifest),
        arrays=MappingProxyType(frozen),
    )


def write_scenario_bundle(
    bundle: CompiledScenarioBundle,
    output_dir: str | Path,
) -> Path:
    """Write a compiled bundle to a new directory without overwriting data."""
    if not isinstance(bundle, CompiledScenarioBundle):
        raise TypeError("bundle must be a CompiledScenarioBundle")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"scenario bundle output already exists: {output}")
    output.mkdir(parents=True)
    for name, value in bundle.arrays.items():
        np.save(output / f"{name}.npy", value, allow_pickle=False)
    (output / "manifest.json").write_text(
        json.dumps(dict(bundle.manifest), allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    )
    return output


def load_scenario_bundle(input_dir: str | Path) -> CompiledScenarioBundle:
    """Load and verify a scenario bundle before a runtime consumes it."""
    root = Path(input_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    payload = require_bound_payload(manifest, label="scenario bundle")
    _require_manifest(payload)
    arrays: dict[str, np.ndarray] = {}
    descriptors = payload["arrays"]
    for name, fields in _ARRAY_FIELDS.items():
        descriptor = descriptors[name]
        value = np.load(root / descriptor["file"], allow_pickle=False)
        value = _freeze_array(value)
        _require_array(name, value, descriptor, fields)
        arrays[name] = value
    return CompiledScenarioBundle(
        manifest=MappingProxyType(dict(manifest)),
        arrays=MappingProxyType(arrays),
    )


def _require_inputs(vehicle, terrain, sensor, mission) -> None:
    expected = (
        (vehicle, SixDofPhysicsProfile, "vehicle"),
        (terrain, BoxRoom, "terrain"),
        (sensor, SixDofSensorProfile, "sensor"),
        (mission, ResolvedMissionPlan, "mission"),
    )
    for value, value_type, label in expected:
        if not isinstance(value, value_type):
            raise TypeError(f"{label} must be a {value_type.__name__}")


def _require_mission_inside_terrain(
    mission: ResolvedMissionPlan,
    terrain: BoxRoom,
) -> None:
    targets = np.asarray(
        [step.target_xyz_m for step in mission.steps],
        dtype=np.float32,
    )
    if not np.all(terrain.contains(targets, margin=0.0)):
        raise ValueError("resolved mission target is outside navigable terrain")


def _freeze_array(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype="<f4")
    if not np.isfinite(result).all():
        raise ValueError("scenario arrays must contain finite float32 values")
    result.setflags(write=False)
    return result


def _array_manifest(
    name: str,
    value: np.ndarray,
    fields: tuple[str, ...],
) -> dict[str, object]:
    return {
        "file": f"{name}.npy",
        "dtype": "float32_le",
        "shape": list(value.shape),
        "fields": list(fields),
        "sha256": sha256_bytes(value.tobytes(order="C")),
    }


def _require_manifest(payload: dict[str, object]) -> None:
    if payload.get("schema") != SCENARIO_BUNDLE_SCHEMA:
        raise ValueError("scenario bundle schema is incompatible")
    if (
        payload.get("authority") != "simulation_only"
        or payload.get("deployment_authority") is not False
        or payload.get("frames") != FRAME_CONTRACT
    ):
        raise ValueError("scenario bundle authority or frame contract is invalid")
    descriptors = payload.get("arrays")
    if not isinstance(descriptors, dict) or set(descriptors) != set(_ARRAY_FIELDS):
        raise ValueError("scenario bundle array manifest is incompatible")


def _require_array(
    name: str,
    value: np.ndarray,
    descriptor: object,
    fields: tuple[str, ...],
) -> None:
    if not isinstance(descriptor, dict):
        raise ValueError(f"scenario array descriptor {name!r} is invalid")
    expected = {
        "file": f"{name}.npy",
        "dtype": "float32_le",
        "shape": list(value.shape),
        "fields": list(fields),
        "sha256": sha256_bytes(value.tobytes(order="C")),
    }
    if descriptor != expected:
        raise ValueError(f"scenario array {name!r} metadata or SHA-256 is invalid")
