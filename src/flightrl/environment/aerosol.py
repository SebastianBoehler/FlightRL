"""Spatial concentration, Beer-Lambert extinction and lit particle scattering."""

import numpy as np
from flightrl.sixdof.geometry import quat_to_matrix


class AerosolCamera:
    def __init__(self, profile, room, boxes=(), width=256, height=192):
        self.profile, self.room = profile, room
        self.width, self.height = width, height
        self.shape = np.ceil((room[1:6:2] - room[:6:2]) / 0.5).astype(int)
        self.cell = (room[1:6:2] - room[:6:2]) / self.shape
        rows, cols = np.mgrid[:height, :width]
        fy = np.tan(np.deg2rad(63) / 2)
        rays = np.stack(
            (
                np.ones_like(cols),
                -(2 * (cols + 0.5) / width - 1) * fy * 4 / 3,
                -(2 * (rows + 0.5) / height - 1) * fy,
            ),
            axis=-1,
        )
        self.rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
        self.mean_transmission = 1.0
        self.set_obstacles(boxes)

    def set_obstacles(self, boxes):
        from .lighting import volume_lighting

        grid = np.moveaxis(np.indices(self.shape), 0, -1)
        points = self.room[:6:2] + (grid.reshape(-1, 3) + 0.5) * self.cell
        self.illumination = volume_lighting(
            self.profile, points, boxes, self.room
        ).reshape(*self.shape, 3)

    def concentration(self, particles):
        grid = np.zeros(self.shape)
        ix = np.clip(
            (
                (particles.position[particles.active] - self.room[:6:2]) / self.cell
            ).astype(int),
            0,
            self.shape - 1,
        )
        np.add.at(grid, tuple(ix.T), 1)
        # Conserved tracer weight; edge padding avoids wraparound at room boundaries.
        for axis in range(3):
            pad = [(0, 0)] * 3
            pad[axis] = (1, 1)
            padded = np.pad(grid, pad, mode="edge")
            a = [slice(None)] * 3
            b = a.copy()
            c = a.copy()
            a[axis] = slice(0, -2)
            b[axis] = slice(1, -1)
            c[axis] = slice(2, None)
            grid = (padded[tuple(a)] + 2 * padded[tuple(b)] + padded[tuple(c)]) / 4
        return grid / (self.profile.particle_count / np.prod(self.shape))

    def apply(self, rgb, depth, position, quaternion, particles):
        if not particles.active.any():
            self.mean_transmission = 1.0
            return
        rotation = quat_to_matrix(np.asarray(quaternion)[None])[0]
        origin = position + rotation @ np.array([0.035, 0, 0.012])
        rays = self.rays @ rotation.T
        grid = self.concentration(particles)
        transmission = np.ones(depth.shape)
        scattering = np.zeros(rgb.shape)
        # Front-to-back Beer-Lambert quadrature with shadowed scene illumination.
        for step in range(12):
            point = origin + rays * (depth * (step + 0.5) / 12)[..., None]
            ix = np.clip(
                ((point - self.room[:6:2]) / self.cell).astype(int), 0, self.shape - 1
            )
            indices = tuple(ix.reshape(-1, 3).T)
            density = grid[indices].reshape(depth.shape)
            absorb = np.exp(-self.profile.dust_extinction_per_m * density * depth / 12)
            light = self.illumination[indices].reshape(rgb.shape)
            scattering += (transmission * (1 - absorb))[..., None] * light
            transmission *= absorb
        self.mean_transmission = float(transmission.mean())
        rgb[:] = np.clip(rgb * transmission[..., None] + scattering, 0, 255)
        local = (particles.position[particles.active] - origin) @ rotation
        for point in local[local[:, 0] > 0.15]:
            x = int(
                (1 - point[1] / point[0] / np.tan(np.deg2rad(63) / 2) / (4 / 3))
                * self.width
                / 2
            )
            y = int(
                (1 - point[2] / point[0] / np.tan(np.deg2rad(63) / 2)) * self.height / 2
            )
            if (
                1 <= x < self.width - 1
                and 1 <= y < self.height - 1
                and np.linalg.norm(point) < depth[y, x]
            ):
                world = origin + rotation @ point
                cell = np.clip(
                    ((world - self.room[:6:2]) / self.cell).astype(int),
                    0,
                    self.shape - 1,
                )
                tint = self.illumination[tuple(cell)]
                # Soft projected parcels supplement the shared concentration volume.
                # Depth-test every covered pixel so dust cannot paint over nearer solids.
                radius = min(
                    14,
                    max(
                        2,
                        int(
                            0.045
                            * self.height
                            / (2 * np.tan(np.deg2rad(63) / 2))
                            / point[0]
                        ),
                    ),
                )
                x0, x1 = max(0, x - radius), min(self.width, x + radius + 1)
                y0, y1 = max(0, y - radius), min(self.height, y + radius + 1)
                yy, xx = np.mgrid[y0:y1, x0:x1]
                alpha = 0.12 * np.exp(-3 * ((xx - x) ** 2 + (yy - y) ** 2) / radius**2)
                alpha *= np.linalg.norm(point) < depth[y0:y1, x0:x1]
                patch = rgb[y0:y1, x0:x1]
                patch[:] = patch * (1 - alpha[..., None]) + tint * alpha[..., None]
