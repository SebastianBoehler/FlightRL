"""Immutable authored inspection geometry, separately bound to a scenario v1."""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from flightrl import _binding
from flightrl.artifact_identity import bind_payload
from flightrl.scenario_bundle import CompiledScenarioBundle, _array_manifest

CAMERA = {
    "shape_hwc": [48, 64, 3],
    "encoding": "rgb_uint8",
    "vertical_fov_rad": 1.099557429,
    "body_offset_m": [0.035, 0, 0.012],
    "optical_axes": "forward=body_x,right=-body_y,down=-body_z",
    "ray_max_distance_m": 8,
    "materials": "diagnostic_solid_rgb_markers_neutral_background",
}
PANEL_FIELDS = tuple(
    "cx cy cz ux uy uz vx vy vz half_width half_height red green blue".split()
)


@dataclass(frozen=True)
class InspectionScene:
    scenario: CompiledScenarioBundle
    panels: np.ndarray
    evaluator_ids: tuple[int, ...]
    manifest: object
    environment: object = None


def compile_inspection_scene(scenario, panels, evaluator_ids, *, environment=None):
    if not isinstance(scenario, CompiledScenarioBundle):
        raise TypeError("scenario must be compiled")
    panels = np.array(panels, dtype="<f4", order="C", copy=True)
    if (
        panels.ndim != 2
        or panels.shape[1] != 14
        or not 0 <= len(panels) <= 1024
        or not np.isfinite(panels).all()
    ):
        raise ValueError("panels require finite (P,14) rows")
    ids = tuple(evaluator_ids)
    if (
        len(ids) != len(panels)
        or len(set(ids)) != len(ids)
        or any(type(i) is not int for i in ids)
    ):
        raise ValueError("unique integer evaluator IDs required")
    u, v = panels[:, 3:6], panels[:, 6:9]
    if not (
        np.allclose(np.linalg.norm(u, axis=1), 1)
        and np.allclose(np.linalg.norm(v, axis=1), 1)
        and np.allclose((u * v).sum(1), 0)
    ):
        raise ValueError("panel axes must be orthonormal")
    if np.any(panels[:, 9:11] <= 0):
        raise ValueError("panel extents must be positive")
    rgb = panels[:, 11:14]
    if np.any((rgb < 0) | (rgb > 255) | (rgb != np.floor(rgb))):
        raise ValueError("panel appearance must be byte RGB")
    room = scenario.arrays["terrain_bounds"]
    for su, sv in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        corner = panels[:, :3] + su * u * panels[:, 9:10] + sv * v * panels[:, 10:11]
        if np.any(corner < room[:6:2]) or np.any(corner > room[1:6:2]):
            raise ValueError("panel corners must stay inside room")
    # True immutable storage, including against setflags(write=True).
    panels = np.frombuffer(panels.tobytes(), dtype="<f4").reshape(-1, 14)
    manifest = bind_payload(
        {
            "schema": "flightrl.inspection_scene.v1",
            "authority": "simulation_only",
            "scenario_sha256": scenario.manifest["sha256"],
            "camera": CAMERA,
            "panels": _array_manifest("panels", panels, PANEL_FIELDS),
            "evaluator_ids": list(ids),
            "body_envelope": "swept_axis_aligned_cube_half_extent_0.08m_terminal_contact",
        }
    )
    if environment is not None:
        from flightrl.environment import EnvironmentProfile

        if not isinstance(environment, EnvironmentProfile):
            raise TypeError("environment must be EnvironmentProfile")
        manifest.pop("sha256")
        manifest = bind_payload({**manifest, "environment": environment.report()})
    return InspectionScene(
        scenario, panels, ids, MappingProxyType(manifest), environment
    )


def render_scene(scene, positions, quaternions, frames=None, counts=None):
    """Counts are evaluator-only [visible, unoccluded projected] pixel counts."""
    positions = np.ascontiguousarray(positions, dtype=np.float32)
    quaternions = np.ascontiguousarray(quaternions, dtype=np.float32)
    n = len(positions)
    if (
        positions.shape != (n, 3)
        or quaternions.shape != (n, 4)
        or not np.allclose(np.linalg.norm(quaternions, axis=1), 1, atol=1e-5)
    ):
        raise ValueError("poses require (N,3) positions and unit wxyz quaternions")
    room = scene.scenario.arrays["terrain_bounds"]
    if np.any(positions <= room[:6:2] + 0.08) or np.any(
        positions >= room[1:6:2] - 0.08
    ):
        raise ValueError("camera body envelope must be inside room")
    frames = np.empty((n, 48, 64, 3), np.uint8) if frames is None else frames
    counts = np.empty((n, len(scene.panels), 2), np.int32) if counts is None else counts
    _binding.inspection_render(
        positions,
        quaternions,
        room,
        scene.scenario.arrays["terrain_obstacles"],
        scene.panels,
        frames,
        counts,
    )
    return frames, counts


def swept_collision(scene, start, end, output=None):
    start = np.ascontiguousarray(start, dtype=np.float32)
    end = np.ascontiguousarray(end, dtype=np.float32)
    output = np.empty(len(start), np.uint8) if output is None else output
    _binding.inspection_collision(
        start,
        end,
        scene.scenario.arrays["terrain_bounds"],
        scene.scenario.arrays["terrain_obstacles"],
        0.08,
        output,
    )
    return output
