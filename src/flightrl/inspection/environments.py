"""Seeded geometry families sharing one RGB-D inspection task and world units."""

from dataclasses import replace
import numpy as np
from flightrl.environment import EnvironmentProfile
from flightrl.inspection.industrial import utility_plant
from flightrl.inspection_scene import compile_inspection_scene
from flightrl.scenario_bundle import compile_scenario_bundle
from flightrl.sixdof.geometry import BoxRoom, AxisAlignedObstacle
from flightrl.sixdof.physics import SixDofPhysicsProfile
from flightrl.sixdof.sensor_model import SixDofSensorProfile
from flightrl.navigation.mission_spec import ResolvedMissionPlan


def environment_scene(family, seed):
    rng = np.random.default_rng(seed)
    base = utility_plant(seed)
    if family == "utility_plant":
        # Change equipment positions in both axes, not only surface randomization.
        arrays = dict(base.scenario.arrays)
        boxes = arrays["terrain_obstacles"].astype(float).copy()
        boxes[:, 5] = np.minimum(boxes[:, 5], 3.4)
        for index in range(4, 10):
            shift = rng.uniform(-0.15, 0.15, 2)
            boxes[index, :4] += np.repeat(shift, 2)
        arrays["terrain_obstacles"] = boxes
        scenario = compile_scenario_bundle(
            vehicle=SixDofPhysicsProfile(),
            terrain=BoxRoom(
                -4, 8, -3, 3, 0, 3.4, 8, tuple(AxisAlignedObstacle(*b) for b in boxes)
            ),
            sensor=SixDofSensorProfile(),
            mission=ResolvedMissionPlan(
                source_text="utility plant exploration", steps=()
            ),
        )
        profile = replace(
            base.environment,
            particle_count=128,
            settled_fraction=1,
            ambient=float(rng.uniform(0.35, 0.55)),
        )
        return compile_inspection_scene(
            scenario, base.panels, base.evaluator_ids, environment=profile
        )
    if family not in ("data_center", "forest"):
        raise ValueError("unknown environment family")
    boxes = []
    if family == "data_center":
        # Two rack rows, open cross-aisles and overhead cable trays.
        for row_y in (-0.1, 1.6):
            for x in (-2.7, -0.8, 1.2, 3.2, 5.2):
                dx, dy = rng.uniform(-0.18, 0.18, 2)
                boxes.append(
                    (x + dx, x + dx + 0.8, row_y + dy, row_y + dy + 0.65, 0, 2.45)
                )
        boxes.extend(
            [(-3.7, 7.5, 0.3, 0.52, 2.9, 3.02), (-3.7, 7.5, 1.8, 2.02, 2.9, 3.02)]
        )
        profile = EnvironmentProfile(
            family,
            surface_style="data_center",
            particle_count=128,
            settled_fraction=1,
            ambient=float(rng.uniform(0.32, 0.48)),
            equipment_rgb=(28, 35, 43),
            floor_rgb=(125, 132, 139),
            wall_rgb=(181, 189, 195),
            floor_roughness=0.28,
            lights=tuple((x, -0.7, 3.28, 0.85, 0.94, 1, 3) for x in (-2, 1, 4, 7)),
        )
        ceiling = 3.4
    else:
        # Tree positions vary by seed; leave a traversable departure strip.
        for x in np.arange(-3, 7.5, 1.7):
            for y in (-0.1, 1.65):
                dx, dy = rng.uniform(-0.35, 0.35, 2)
                radius = float(rng.uniform(0.10, 0.19))
                cx, cy = float(np.clip(x + dx, -3.3, 7.3)), y + dy
                height = float(rng.uniform(4.0, 5.0))
                boxes.append(
                    (cx - radius, cx + radius, cy - radius, cy + radius, 0, height)
                )
                boxes.append(
                    (cx - 0.7, cx + 0.7, cy - 0.7, cy + 0.7, height - 0.9, height + 1.2)
                )
        for x in (-0.4, 3.7, 6.8):
            boxes.append((x, x + 0.45, -2.7, -2.3, 0, 0.35))
        profile = EnvironmentProfile(
            family,
            surface_style="forest",
            particle_count=128,
            settled_fraction=1,
            ambient=float(rng.uniform(0.5, 0.7)),
            equipment_rgb=(91, 66, 42),
            floor_rgb=(74, 82, 47),
            wall_rgb=(155, 190, 213),
            sun_strength=1.4,
            sun_direction=(0.35, 0.45, 0.82),
            wind_m_s=(0.12, 0.04, 0),
            turbulence_m_s=0.08,
        )
        ceiling = 7.0
    scenario = compile_scenario_bundle(
        vehicle=SixDofPhysicsProfile(),
        terrain=BoxRoom(
            -4, 8, -3, 3, 0, ceiling, 8, tuple(AxisAlignedObstacle(*b) for b in boxes)
        ),
        sensor=SixDofSensorProfile(),
        mission=ResolvedMissionPlan(source_text=f"{family} exploration", steps=()),
    )
    return compile_inspection_scene(
        scenario, base.panels, base.evaluator_ids, environment=profile
    )
