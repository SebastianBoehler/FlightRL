"""Owned three-marker room for inspection diagnostics; not a navigation policy."""

import numpy as np
from flightrl.inspection_scene import compile_inspection_scene
from flightrl.navigation.mission_spec import ResolvedMissionPlan
from flightrl.scenario_bundle import compile_scenario_bundle
from flightrl.sixdof.geometry import AxisAlignedObstacle, BoxRoom
from flightrl.sixdof.physics import SixDofPhysicsProfile
from flightrl.sixdof.sensor_model import SixDofSensorProfile


def three_panel_room():
    scenario = compile_scenario_bundle(
        vehicle=SixDofPhysicsProfile(),
        terrain=BoxRoom(
            x_min=-4,
            x_max=4,
            y_min=-3,
            y_max=3,
            z_min=0,
            z_max=3,
            max_range_m=8,
            obstacles=(AxisAlignedObstacle(2, 2.2, -0.55, 0.55, 0, 2.5),),
        ),
        sensor=SixDofSensorProfile(),
        mission=ResolvedMissionPlan(
            source_text="three-panel diagnostic; no navigation", steps=()
        ),
    )
    panels = np.array(
        [
            [3.99, y, 1.5, 0, -1, 0, 0, 0, 1, 0.45, 0.45, *rgb]
            for y, rgb in (
                (-1.5, (220, 30, 30)),
                (0, (30, 220, 30)),
                (1.5, (30, 30, 220)),
            )
        ],
        np.float32,
    )
    return compile_inspection_scene(scenario, panels, (101, 102, 103))
