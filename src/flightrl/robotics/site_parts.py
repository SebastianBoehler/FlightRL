"""Shared physical primitives for an authored industrial utility campus."""

import numpy as np

STEEL = [0.23, 0.31, 0.34]
PIPE = [0.51, 0.57, 0.59]
YELLOW = [0.95, 0.57, 0.08]


class Parts:
    def __init__(self):
        self.items = []

    def box(self, name, p, half, color=STEEL, yaw=0, **extra):
        self.items.append(
            dict(
                name=name,
                position=list(p),
                half_extents=list(half),
                color=[*color, 1],
                quaternion=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)],
                **extra,
            )
        )

    def cylinder(self, name, a, b, radius, color=PIPE):
        a, b = np.array(a, float), np.array(b, float)
        direction = b - a
        length = np.linalg.norm(direction)
        if length < 0.001:
            raise ValueError("Nonzero pipe length required")
        axis = direction / length
        q = np.r_[1 + axis[2], -axis[1], axis[0], 0]
        q = (
            np.array([0, 1, 0, 0])
            if np.linalg.norm(q) < 1e-8
            else q / np.linalg.norm(q)
        )
        self.items.append(
            dict(
                name=name,
                type="cylinder",
                position=((a + b) / 2).tolist(),
                size=[radius, length / 2],
                quaternion=q.tolist(),
                color=[*color, 1],
            )
        )

    def transformer(self, name, x, y, scale=1):
        self.box(
            name + "_pad",
            [x, y, 0.12],
            [1.35 * scale, 1.15 * scale, 0.12],
            [0.38, 0.4, 0.39],
        )
        self.box(
            name + "_tank",
            [x, y, 1.1 * scale],
            [0.85 * scale, 0.65 * scale, 0.85 * scale],
            [0.3, 0.4, 0.38],
        )
        for side in (-1, 1):
            for j in range(10):
                self.box(
                    f"{name}_fin_{side}_{j}",
                    [
                        x - 0.8 * scale + j * 0.175 * scale,
                        y + side * 0.88 * scale,
                        1.05 * scale,
                    ],
                    [0.025, 0.23 * scale, 0.65 * scale],
                    PIPE,
                )
        for j in range(3):
            xx = x + (j - 1) * 0.55 * scale
            self.cylinder(
                f"{name}_bushing_{j}",
                [xx, y, 1.9 * scale],
                [xx, y, 2.85 * scale],
                0.09 * scale,
                [0.31, 0.22, 0.15],
            )
            for k in range(6):
                self.cylinder(
                    f"{name}_insulator_{j}_{k}",
                    [xx, y, (2 + k * 0.12) * scale],
                    [xx, y, (2.045 + k * 0.12) * scale],
                    0.16 * scale,
                    [0.5, 0.36, 0.23],
                )

    def pump(self, name, x, y):
        self.box(name + "_base", [x, y, 0.13], [0.9, 0.46, 0.13], [0.31, 0.33, 0.32])
        self.cylinder(
            name + "_motor",
            [x - 0.65, y, 0.52],
            [x + 0.15, y, 0.52],
            0.28,
            [0.1, 0.34, 0.38],
        )
        for j in range(7):
            self.cylinder(
                f"{name}_fin{j}",
                [x - 0.6 + j * 0.09, y, 0.52],
                [x - 0.58 + j * 0.09, y, 0.52],
                0.31,
                [0.19, 0.39, 0.41],
            )
        self.cylinder(
            name + "_volute",
            [x + 0.15, y, 0.52],
            [x + 0.5, y, 0.52],
            0.34,
            [0.18, 0.43, 0.45],
        )
        self.cylinder(name + "_riser", [x + 0.34, y, 0.52], [x + 0.34, y, 1.5], 0.09)
        self.cylinder(
            name + "_valve",
            [x + 0.34, y, 1.2],
            [x + 0.34, y + 0.23, 1.2],
            0.055,
            YELLOW,
        )
        self.cylinder(
            name + "_wheel",
            [x + 0.34, y + 0.23, 1.2],
            [x + 0.34, y + 0.27, 1.2],
            0.17,
            YELLOW,
        )

    def rack(self, name, x, length=16):
        for y in (-7, -1, 5, 11):
            for xx in (x - 0.65, x + 0.65):
                self.box(f"{name}_post_{xx}_{y}", [xx, y, 2.1], [0.07, 0.07, 2.1])
            self.box(f"{name}_cross_{y}", [x, y, 4.15], [0.85, 0.07, 0.07])
        for j, color in enumerate(([0.37, 0.5, 0.52], PIPE, YELLOW)):
            xx = x + (j - 1) * 0.43
            self.cylinder(
                f"{name}_pipe{j}", [xx, -8, 4.35], [xx, -8 + length, 4.35], 0.1, color
            )

    def production_cell(self, name, x, y):
        self.box(name + "_bed", [x, y, 0.55], [1.05, 0.7, 0.15], [0.19, 0.27, 0.28])
        for dx in (-0.95, 0.95):
            for dy in (-0.6, 0.6):
                self.box(
                    f"{name}_leg{dx}_{dy}", [x + dx, y + dy, 0.25], [0.07, 0.07, 0.25]
                )
        for j in range(12):
            xx = x - 0.9 + j * 0.16
            self.cylinder(
                f"{name}_roller{j}", [xx, y - 0.65, 0.76], [xx, y + 0.65, 0.76], 0.06
            )
        for dx in (-1.1, 1.1):
            self.box(
                f"{name}_post{dx}",
                [x + dx, y, 1.3],
                [0.09, 0.85, 1.3],
                [0.37, 0.48, 0.49],
            )
        self.box(name + "_press", [x, y, 2.5], [1.2, 0.85, 0.15], [0.37, 0.48, 0.49])
        self.box(name + "_tool", [x, y, 1.9], [0.15, 0.2, 0.45], YELLOW)
