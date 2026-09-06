"""Observed RGB-D mapping in the declared odometry frame; no scene access."""

import heapq
import numpy as np
from flightrl.sixdof.geometry import quat_to_matrix

CELL = 0.25
FY = np.tan(1.099557429 / 2)


def camera_points(depth, position, quaternion, stride=4):
    rows, cols = np.mgrid[4:44:stride, 0:64:stride]
    rays = np.stack(
        (
            np.ones_like(cols),
            -(2 * (cols + 0.5) / 64 - 1) * FY * 64 / 48,
            -(2 * (rows + 0.5) / 48 - 1) * FY,
        ),
        axis=-1,
    )
    rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    rotation = quat_to_matrix(quaternion[None])[0]
    origin = position + rotation @ np.array([0.035, 0, 0.012])
    points = origin + (rays * depth[rows, cols, None]) @ rotation.T
    return origin, points.reshape(-1, 3)


class ObservedMap:
    def __init__(self):
        self.free = set()
        self.occupied = set()
        self.visits = {}

    @staticmethod
    def cell(point):
        return tuple(np.floor(np.asarray(point)[:2] / CELL).astype(int))

    @staticmethod
    def center(cell):
        return (np.array(cell) + 0.5) * CELL

    def update(self, depth, position, quaternion):
        origin, points = camera_points(depth, position, quaternion)
        self.visits[self.cell(position)] = self.visits.get(self.cell(position), 0) + 1
        self.free.add(self.cell(position))
        for endpoint, distance_ray in zip(points, depth[4:44:4, ::4].ravel()):
            if not np.isfinite(distance_ray) or distance_ray <= 0:
                continue
            # Horizontal flight slice. Ground/ceiling returns do not become walls.
            if abs(endpoint[2] - position[2]) > 0.40:
                continue
            distance = np.linalg.norm(endpoint[:2] - origin[:2])
            steps = max(2, int(distance / (CELL * 0.6)))
            cells = {self.cell(p) for p in np.linspace(origin, endpoint, steps)[:-1]}
            self.free.update(cells)
            # Eight metres is the native renderer's range limit, not a surface.
            if distance_ray < 8:
                self.occupied.add(self.cell(endpoint))
        self.free.difference_update(self.occupied)

    def safe(self):
        blocked = set()
        for x, y in self.occupied:
            blocked.update((x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
        return self.free - blocked

    def path(self, start, goal):
        start = self.cell(start)
        goal = self.cell(goal)
        safe = self.safe() | {start}
        if goal not in safe:
            candidates = [c for c in safe if np.linalg.norm(np.array(c) - goal) <= 2]
            if not candidates:
                return []
            goal = min(candidates, key=lambda c: np.linalg.norm(np.array(c) - goal))
        queue = [(0, start)]
        parents = {start: None}
        costs = {start: 0}
        while queue:
            _, current = heapq.heappop(queue)
            if current == goal:
                route = []
                while current is not None:
                    route.append(self.center(current))
                    current = parents[current]
                return route[::-1]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (current[0] + dx, current[1] + dy)
                new = costs[current] + 1
                if nxt in safe and new < costs.get(nxt, float("inf")):
                    costs[nxt] = new
                    parents[nxt] = current
                    heapq.heappush(
                        queue,
                        (new + abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1]), nxt),
                    )
        return []

    def frontier(self, position):
        safe = self.safe()
        candidates = [
            c
            for c in safe
            if any(
                (c[0] + dx, c[1] + dy) not in self.free | self.occupied
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            )
        ]
        candidates = [
            c for c in candidates if np.linalg.norm(self.center(c) - position[:2]) > 0.6
        ]
        return sorted(
            candidates,
            key=lambda c: (
                np.linalg.norm(self.center(c) - position[:2])
                + self.visits.get(c, 0) * 0.15
            ),
        )
