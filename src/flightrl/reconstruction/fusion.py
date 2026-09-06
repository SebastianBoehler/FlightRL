"""Empty-start voxel-sampled surface map; first observations retain capture time."""

import numpy as np
from .geometry import dense_points, transform


class SurfaceMap:
    def __init__(self, cell=0.08):
        self.cell = cell
        self.voxels = {}

    def integrate(self, rgb, depth, k, pose, frame):
        points, pixels = dense_points(depth, k)
        points = transform(points, pose)
        self.add(points, rgb[pixels[:, 1], pixels[:, 0]], frame)

    def add(self, points, colors, frame):
        points, colors = np.asarray(points), np.asarray(colors)
        if not len(points):
            return
        valid = np.isfinite(points).all(axis=1)
        points, colors = points[valid], colors[valid]
        keys = np.floor(points / self.cell).astype(np.int64).tolist()
        for cell, point, color in zip(keys, points, colors):
            key = tuple(cell)
            if key not in self.voxels:
                self.voxels[key] = (point.tolist(), color.tolist(), frame)

    def export(self):
        return list(self.voxels.values())
