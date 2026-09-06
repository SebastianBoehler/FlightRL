"""Owned utility plant: equipment rooms, service doors and conservative pipe bounds."""

import numpy as np
from flightrl.inspection_scene import compile_inspection_scene
from flightrl.scenario_bundle import compile_scenario_bundle
from flightrl.navigation.mission_spec import ResolvedMissionPlan
from flightrl.sixdof.geometry import AxisAlignedObstacle, BoxRoom
from flightrl.sixdof.physics import SixDofPhysicsProfile
from flightrl.sixdof.sensor_model import SixDofSensorProfile
from flightrl.inspection.controller import MissionController


class IndustrialMission(MissionController):
    scan_ticks = 110
    waypoint_tolerance = 0.10

    def __init__(self, start):
        super().__init__(start)
        self.scan_anchor = None

    def command(self, position, quaternion):
        command, goal = super().command(position, quaternion)
        if self.mode == "scan" and not self.finished:
            if self.scan_anchor is None:
                self.scan_anchor = position[:2].copy()
            yaw = 2 * np.arctan2(quaternion[3], quaternion[0])
            delta = self.scan_anchor - position[:2]
            c, s = np.cos(yaw), np.sin(yaw)
            body = np.array([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]])
            command[:2] = np.clip(body * 1.8, -0.3, 0.3)
            goal[:2] = body
        else:
            self.scan_anchor = None
        return command, goal

    def observe(self, rgb, *args):
        signal = rgb.astype(np.float32)
        normalized = np.clip(
            signal * (220 / np.maximum(signal.max(axis=2, keepdims=True), 80)), 0, 255
        ).astype(np.uint8)
        super().observe(normalized, *args)


def utility_plant(seed=0, *, heavy_dust=False):
    rng = np.random.default_rng(seed)
    shift = float(rng.uniform(-0.12, 0.12))
    boxes = [
        (-0.10, 0.10, -3, -1.65, 0, 3.4),
        (-0.10, 0.10, -0.05, 3, 0, 3.4),
        (3.9, 4.1, -3, 0.35, 0, 3.4),
        (3.9, 4.1, 2.05, 3, 0, 3.4),
        (-3.5, -2.5, 1, 2.4, 0, 1.9),
        (-1.1, -0.45, 1.5, 2.6, 0, 2.3),
        (1.1, 1.75, -0.3 + shift, 1.25 + shift, 0, 2.0),
        (2.5, 3.35, -2.65, -1.4, 0, 1.8),
        (5.2, 6.1, -2.5, -1, 0, 2.1),
        (6.5, 7.3, 1.9, 2.6, 0, 1.6),
        (-3.7, 7.6, 2.65, 2.82, 2.55, 2.72),
        (-3.7, 7.6, 2.3, 2.47, 2.8, 2.97),
        (2.65, 2.82, -2.5, 2.75, 2.8, 2.97),
    ]
    bundle = compile_scenario_bundle(
        vehicle=SixDofPhysicsProfile(),
        terrain=BoxRoom(
            -4, 8, -3, 3, 0, 3.4, 8, tuple(AxisAlignedObstacle(*b) for b in boxes)
        ),
        sensor=SixDofSensorProfile(),
        mission=ResolvedMissionPlan(source_text="utility plant exploration", steps=()),
    )
    panels = np.array(
        [
            [-1.5, 2.99, 1.5, 1, 0, 0, 0, 0, 1, 0.45, 0.45, 220, 30, 30],
            [2, -2.99, 1.5, -1, 0, 0, 0, 0, 1, 0.45, 0.45, 30, 220, 30],
            [7.99, 0.7, 1.5, 0, -1, 0, 0, 0, 1, 0.45, 0.45, 30, 30, 220],
        ],
        np.float32,
    )
    from flightrl.environment import EnvironmentProfile

    environment = EnvironmentProfile(
        name="utility_plant_heavy_dust" if heavy_dust else "utility_plant",
        particle_count=4096,
        settled_fraction=0.8,
        dust_extinction_per_m=0.45 if heavy_dust else 0.035,
        lights=tuple((x, -0.15, 3.28, 1.0, 1.0, 1.0, 2.5) for x in (-2.0, 2.0, 6.0)),
    )
    return compile_inspection_scene(
        bundle, panels, (101, 102, 103), environment=environment
    )
