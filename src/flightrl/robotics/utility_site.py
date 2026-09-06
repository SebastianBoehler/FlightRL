"""Seeded power/production utility sites with task-bearing equipment and real geometry."""

import numpy as np
from .site_parts import Parts, STEEL, PIPE, YELLOW


def utility_site(seed=0):
    rng = np.random.default_rng(seed)
    parts = Parts()
    variant = seed % 2
    lane = float(rng.uniform(-0.55, 0.55))
    length = float(rng.uniform(2.8, 4.6))
    selected = int(rng.integers(3))
    target_y = [-1.8 + lane, 1.8 + lane]
    signal = int(rng.integers(2))
    # The whole layout, equipment types, aisle spacing and target placement vary by seed.
    for side in (-1, 1):
        for i in range(4):
            x = -6 + i * 4 + float(rng.uniform(-0.45, 0.45))
            y = side * (6 + float(rng.uniform(-0.4, 0.4)))
            if variant == 1 and side == -1:
                parts.production_cell(f"production_{i}", x, y)
            elif (i + variant) % 2 == 0:
                parts.transformer(
                    f"transformer_{side}_{i}", x, y, float(rng.uniform(0.8, 1.1))
                )
            else:
                parts.pump(f"pump_{side}_{i}", x, y)
    for x in (-7, 8):
        parts.rack(f"pipe_rack{x}", x)
    # Production hall around the rear service area: high roof, windows and overhead gantry.
    for x in (-8, -3, 2, 7, 12):
        for y in (-10, 10):
            parts.box(f"hall_column{x}_{y}", [x, y, 3.1], [0.12, 0.12, 3.1])
        parts.box(f"roof_beam{x}", [x, 0, 6.25], [0.11, 10.15, 0.11])
    for side in (-1, 1):
        parts.box(
            f"wall_base{side}", [2, side * 10.2, 1], [10, 0.1, 1], [0.36, 0.41, 0.4]
        )
        parts.box(
            f"wall_cladding{side}",
            [2, side * 10.2, 4.65],
            [10, 0.1, 1.5],
            [0.51, 0.59, 0.58],
        )
        for i in range(32):
            parts.box(
                f"wall_rib{side}_{i}",
                [-7.8 + i * 0.62, side * 10.05, 4.65],
                [0.018, 0.04, 1.5],
                [0.36, 0.44, 0.45],
            )
    parts.box("gantry_beam", [7, 0, 5.65], [0.18, 9.5, 0.24], YELLOW)
    parts.box("hoist", [7, 3.2, 5.15], [0.36, 0.4, 0.27], [0.75, 0.38, 0.04])
    parts.cylinder("hoist_cable", [7, 3.2, 4.9], [7, 3.2, 3.8], 0.025, STEEL)
    # Remote yard silhouettes are modeled geometry, too.
    for i in range(3):
        x = 17 + i * 5
        parts.cylinder(f"tank{i}", [x, 8, 0.2], [x, 8, 4.5], 1.5, [0.55, 0.61, 0.6])
        parts.cylinder(f"stack{i}", [x, -9, 0], [x, -9, 9 + i], 0.5, [0.38, 0.43, 0.44])
    for y in (-13, 13):
        for i in range(17):
            parts.box(
                f"fence_{y}_{i}", [-10 + i * 2.5, y, 1.1], [0.035, 0.035, 1.1], PIPE
            )
        for z in (0.5, 1.2, 2):
            parts.cylinder(f"fence_rail{y}_{z}", [-10, y, z], [30, y, z], 0.025, PIPE)
    # Clearly separated inspection lanes and a low pipe over the rover route.
    for y in (-3.25, 0, 3.25):
        for x in np.arange(-5, 7, 0.8):
            parts.box(
                f"lane_paint{x}_{y}", [float(x), y, 0.004], [0.25, 0.035, 0.003], YELLOW
            )
    parts.cylinder(
        "service_pipe",
        [-0.4, target_y[1] - 0.8, 1.15],
        [-0.4, target_y[1] + 0.8, 1.15],
        0.14,
    )
    targets = []
    for i, (ident, z, size, robot) in enumerate(
        (
            (17 + selected, 1.85 + float(rng.uniform(-0.25, 0.25)), 0.7, "drone"),
            (23 + selected, 0.34, 0.55, "rover"),
        )
    ):
        x = length + float(rng.uniform(-0.2, 0.2))
        y = target_y[i]
        target = dict(
            id=ident,
            robot=robot,
            position=[x, y, z],
            size=size,
            approach=[-1, 0, 0],
            asset=f"{'Transformer' if i == 0 else 'Pump'} {selected + 1:02}",
            signal=signal,
        )
        targets.append(target)
        # A real cabinet fascia supports the tag, indicator and inspection point.
        parts.box(
            f"inspection_cabinet_{i}",
            [x + 0.4, y, z / 2],
            [0.38, 0.6, z / 2],
            [0.25, 0.36, 0.37],
        )
        parts.box(
            f"marker_{ident}",
            [x, y, z],
            [0.018, size / 2, size / 2],
            [0.96, 0.96, 0.93],
        )
        parts.box(
            f"signal_{ident}",
            [x - 0.025, y + size * 0.7, z],
            [0.025, 0.055, 0.055],
            [0.95, 0.035, 0.012] if signal else [0.015, 0.75, 0.12],
        )
        for j in range(4):
            parts.box(
                f"panel_vent{i}_{j}",
                [x - 0.005, y - 0.43, z * 0.4 + j * 0.06],
                [0.01, 0.08, 0.014],
                [0.09, 0.13, 0.14],
            )
    targets.append(
        dict(
            id=42,
            robot="rover",
            position=[-4.4, target_y[1], 0.34],
            size=0.55,
            approach=[1, 0, 0],
            asset="Charging dock",
            signal=None,
        )
    )
    parts.box(
        "marker_42", targets[-1]["position"], [0.018, 0.275, 0.275], [0.96, 0.96, 0.93]
    )
    parts.box(
        "dock_platform",
        [-4.05, target_y[1], 0.018],
        [0.35, 0.48, 0.018],
        [0.17, 0.24, 0.24],
    )
    # Seeded task and appearance metadata accompanies the physical geometry.
    return (
        parts.items,
        targets,
        dict(
            name="Production utilities" if variant else "Power utility campus",
            variant=variant,
            seed=seed,
            spawns=[
                [-3, target_y[0] + float(rng.uniform(-0.3, 0.3)), 1.75],
                [-3, target_y[1] + float(rng.uniform(-0.25, 0.25)), 0.17],
            ],
            yaw=float(rng.uniform(-0.15, 0.15)),
            friction=float(rng.uniform(0.48, 0.95)),
            wind=[float(rng.uniform(-0.08, 0.08)), float(rng.uniform(-0.08, 0.08)), 0],
            sun_angle=float(rng.uniform(-0.3, 0.3)),
            haze=float(rng.uniform(0.004, 0.014)),
        ),
    )
