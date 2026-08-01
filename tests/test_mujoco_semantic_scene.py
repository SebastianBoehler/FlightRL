from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from flightrl.mujoco import is_mujoco_rendering_available
from flightrl.mujoco.env import MuJoCoCrazyflieEnv
from flightrl.mujoco.model import build_crazyflie_mjcf
from flightrl.mujoco.semantic_reset import (
    SEMANTIC_RESET_CLEARANCE_M,
    sample_collision_free_reset,
)
from flightrl.navigation import Bounds3D, SemanticObject, SemanticScene, TargetAnchor
from flightrl.sixdof.curriculum import resolve_reset_profile
from flightrl.sixdof.geometry import AxisAlignedObstacle, BoxRoom


def test_semantic_objects_are_named_renderable_mujoco_geometries() -> None:
    scene = _scene()

    root = ET.fromstring(build_crazyflie_mjcf(scene))
    desk = root.find(".//geom[@name='semantic_desk']")
    door = root.find(".//geom[@name='semantic_door']")

    assert desk is not None
    assert desk.attrib["pos"] == "0 0.7 0.4"
    assert desk.attrib["size"] == "0.5 0.3 0.4"
    assert desk.attrib["contype"] == "1"
    assert door is not None
    assert door.attrib["contype"] == "0"
    assert root.find(".//camera[@name='aideck']") is not None


def test_door_rendering_has_noncolliding_structural_cues() -> None:
    root = ET.fromstring(build_crazyflie_mjcf(_scene()))
    names = (
        "semantic_door_frame_left",
        "semantic_door_frame_right",
        "semantic_door_frame_top",
        "semantic_door_panel_upper",
        "semantic_door_handle",
    )

    geoms = [root.find(f".//geom[@name='{name}']") for name in names]

    assert all(geom is not None for geom in geoms)
    assert all(
        geom.attrib["contype"] == "0"
        for geom in geoms
        if geom is not None
    )


def test_mujoco_backend_loads_semantic_geometry_and_range_colliders() -> None:
    env = MuJoCoCrazyflieEnv(num_envs=1, semantic_scene=_scene())
    try:
        geom_id = env.mujoco.mj_name2id(
            env.model,
            env.mujoco.mjtObj.mjOBJ_GEOM,
            "semantic_desk",
        )
        assert geom_id >= 0
        assert len(env.room.obstacles) == 1
    finally:
        del env


def test_mujoco_semantic_scene_renders_aideck_frame_when_available() -> None:
    if not is_mujoco_rendering_available():
        pytest.skip("MuJoCo rendering backend is unavailable")
    env = MuJoCoCrazyflieEnv(num_envs=1, semantic_scene=_scene())
    try:
        assert env.render_aideck_gray4().shape == (48, 64)
    finally:
        del env


def test_semantic_reset_filters_both_starts_and_targets_from_colliders() -> None:
    room = BoxRoom(
        obstacles=(
            AxisAlignedObstacle(
                x_min=-0.25,
                x_max=0.25,
                y_min=-0.25,
                y_max=0.25,
                z_min=0.0,
                z_max=1.2,
            ),
        ),
    )

    reset = sample_collision_free_reset(
        resolve_reset_profile("broad"),
        np.random.default_rng(123),
        10_000,
        room,
        clearance_m=SEMANTIC_RESET_CLEARANCE_M,
    )

    assert room.contains(reset[0], margin=SEMANTIC_RESET_CLEARANCE_M).all()
    assert room.contains(reset[4], margin=SEMANTIC_RESET_CLEARANCE_M).all()


def _scene() -> SemanticScene:
    return SemanticScene(
        room=Bounds3D((-2.0, -2.0, 0.0), (2.0, 2.0, 2.5)),
        objects=(
            SemanticObject(
                object_id="desk",
                category="table",
                bounds=Bounds3D((-0.5, 0.4, 0.0), (0.5, 1.0, 0.8)),
                preferred_anchor=TargetAnchor.APPROACH,
                approach_position_m=(0.0, 0.0, 0.8),
            ),
            SemanticObject(
                object_id="door",
                category="doorway",
                bounds=Bounds3D((1.94, -0.4, 0.0), (1.99, 0.4, 2.1)),
                preferred_anchor=TargetAnchor.APPROACH,
                approach_position_m=(1.5, 0.0, 0.8),
                collision=False,
            ),
        ),
    )
