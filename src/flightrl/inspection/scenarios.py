"""Frozen procedural split: training 0..7, validation 8..9, test 100..119."""

import numpy as np
from flightrl.inspection_scene import compile_inspection_scene
from flightrl.navigation.mission_spec import ResolvedMissionPlan
from flightrl.scenario_bundle import compile_scenario_bundle
from flightrl.sixdof.geometry import BoxRoom, AxisAlignedObstacle
from flightrl.sixdof.physics import SixDofPhysicsProfile
from flightrl.sixdof.sensor_model import SixDofSensorProfile

SPLITS = {"train": list(range(8)), "validation": [8, 9], "test": list(range(100, 120))}
GATES = {
    "min_mean_coverage": 0.9,
    "max_collision_rate": 0.0,
    "min_recovery_rate": 0.9,
    "mission_ticks": 900,
    "policy_frozen_before_test": True,
}


def scenario(seed):
    rng = np.random.default_rng(seed)
    xmax = float(rng.uniform(3.8, 4.5))
    center = float(rng.uniform(-0.25, 0.25))
    ox = float(rng.uniform(1.7, 2.2))
    box = AxisAlignedObstacle(ox, ox + 0.25, center - 0.5, center + 0.5, 0, 2.5)
    bundle = compile_scenario_bundle(
        vehicle=SixDofPhysicsProfile(),
        terrain=BoxRoom(-4, xmax, -3, 3, 0, 3, 8, (box,)),
        sensor=SixDofSensorProfile(),
        mission=ResolvedMissionPlan(
            source_text="bounded authored industrial inspection", steps=()
        ),
    )
    panels = []
    for y, rgb in zip((-1.5, 0, 1.5), ((220, 30, 30), (30, 220, 30), (30, 30, 220))):
        color = np.clip(np.array(rgb) + rng.integers(-12, 13, 3), 0, 255)
        panels.append(
            [
                xmax - 0.01,
                y + float(rng.uniform(-0.15, 0.15)),
                1.5,
                0,
                -1,
                0,
                0,
                0,
                1,
                0.45,
                0.45,
                *color,
            ]
        )
    return compile_inspection_scene(
        bundle, np.array(panels, np.float32), (101, 102, 103)
    )
