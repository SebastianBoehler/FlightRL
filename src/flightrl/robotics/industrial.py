"""Bounded mixed-robot plant inspection geometry and visual target placement."""

import numpy as np


def equipment(seed=0):
    rng = np.random.default_rng(seed)
    items = []

    def box(name, p, h, color):
        items.append(dict(name=name, position=p, half_extents=h, color=[*color, 1]))

    for side in (-1, 1):
        box(f"wall_{side}", [2, side * 4, 1.8], [7, 0.12, 1.8], [0.48, 0.52, 0.52])
        for i in range(6):
            x = -3 + i * 2
            box(
                f"column_{side}_{i}",
                [x, side * 3.6, 1.8],
                [0.1, 0.1, 1.8],
                [0.3, 0.34, 0.36],
            )
            box(f"beam_{side}_{i}", [x, 0, 3.65], [0.09, 3.7, 0.09], [0.3, 0.34, 0.36])
    for i in range(5):
        x = -1 + i * 1.6
        box(f"machine_{i}", [x, 3.1, 0.65], [0.5, 0.45, 0.65], [0.28, 0.39, 0.42])
        box(f"cabinet_{i}", [x, -3.1, 0.8], [0.38, 0.4, 0.8], [0.5, 0.55, 0.51])
        for j in range(3):
            box(
                f"vent_{i}_{j}",
                [x, -2.69, 0.7 + j * 0.12],
                [0.25, 0.015, 0.025],
                [0.15, 0.18, 0.17],
            )
        box(f"pipe_{i}", [x, 2.7, 2.6], [0.06, 0.06, 0.8], [0.55, 0.56, 0.5])
    targets = [
        dict(
            id=17,
            robot="drone",
            position=[3 + float(rng.uniform(-0.2, 0.2)), -1.2, 1.6],
            size=0.65,
        ),
        dict(
            id=23,
            robot="rover",
            position=[3 + float(rng.uniform(-0.2, 0.2)), 1.2, 0.31],
            size=0.55,
        ),
        dict(id=42, robot="rover", position=[-3.3, 1.2, 0.31], size=0.55),
    ]
    for target in targets:
        x, y, z = target["position"]
        size = target["size"]
        box(
            f"marker_{target['id']}",
            [x, y, z],
            [0.015, size * 0.5, size * 0.5],
            [0.96, 0.96, 0.93],
        )
        box(
            f"stand_{target['id']}",
            [x + 0.06, y, (z - size * 0.5) * 0.5],
            [0.04, 0.04, (z - size * 0.5) * 0.5],
            [0.3, 0.32, 0.3],
        )
    # A low service duct above the rover lane, visible and physically collidable.
    box("service_duct", [0.2, 1.2, 1.15], [0.5, 0.55, 0.12], [0.5, 0.53, 0.55])
    return items, targets
