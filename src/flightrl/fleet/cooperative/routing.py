"""Conservative known-map routing for the bounded cooperation experiment."""

import heapq
import numpy as np


class Routes:
    def __init__(self, scene, clearance=0.43, step=0.25):
        self.room = scene.scenario.arrays["terrain_bounds"]
        self.boxes = scene.scenario.arrays["terrain_obstacles"]
        self.step = step
        self.origin = self.room[[0, 2]] + 0.6
        self.shape = tuple(
            np.floor((self.room[[1, 3]] - 0.6 - self.origin) / step).astype(int) + 1
        )
        self.free = set()
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                p = self.point((x, y))
                if not any(
                    b[4] < 2.95
                    and b[5] > 0.55
                    and b[0] - clearance <= p[0] <= b[1] + clearance
                    and b[2] - clearance <= p[1] <= b[3] + clearance
                    for b in self.boxes
                ):
                    self.free.add((x, y))
        self.cache = {}

    def point(self, cell):
        return self.origin + np.array(cell) * self.step

    def cell(self, point):
        cell = tuple(
            np.rint((np.array(point[:2]) - self.origin) / self.step).astype(int)
        )
        if cell not in self.free:
            raise ValueError("Position outside conservative navigable grid")
        return cell

    def path(self, start, end):
        a, b = self.cell(start), self.cell(end)
        key = (a, b)
        if key in self.cache:
            return self.cache[key]
        queue = [(0, a)]
        previous, cost = {}, {a: 0}
        while queue:
            _, p = heapq.heappop(queue)
            if p == b:
                cells = [b]
                while cells[-1] != a:
                    cells.append(previous[cells[-1]])
                cells.reverse()
                # Preserve corners; remove collinear intermediate grid cells.
                short = [cells[0]]
                for i in range(1, len(cells) - 1):
                    if tuple(np.subtract(cells[i], cells[i - 1])) != tuple(
                        np.subtract(cells[i + 1], cells[i])
                    ):
                        short.append(cells[i])
                if len(cells) > 1:
                    short.append(cells[-1])
                result = np.array([self.point(c) for c in short])
                self.cache[key] = result
                return result
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (p[0] + d[0], p[1] + d[1])
                g = cost[p] + 1
                if n in self.free and g < cost.get(n, float("inf")):
                    cost[n], previous[n] = g, p
                    heapq.heappush(queue, (g + abs(n[0] - b[0]) + abs(n[1] - b[1]), n))
        raise ValueError("No connected route")

    def length(self, a, b):
        path = self.path(a, b)
        return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())

    def features(self, a, b):
        a, b = np.array(a[:2]), np.array(b[:2])
        d = b - a
        low, high = np.minimum(a, b), np.maximum(a, b)
        blocking = [
            box
            for box in self.boxes
            if box[4] < 2.95
            and box[5] > 0.55
            and box[0] < high[0]
            and box[1] > low[0]
            and box[2] < high[1]
            and box[3] > low[1]
        ]
        return np.r_[
            a / 8,
            b / 8,
            d / 8,
            np.abs(d) / 8,
            np.linalg.norm(d) / 8,
            len(blocking) / 10,
        ].astype(np.float32)

    def sites(self, seed, count=12):
        rng = np.random.default_rng(seed)
        candidates = sorted(self.free)
        chosen = []
        for idx in rng.permutation(len(candidates)):
            point = self.point(candidates[idx])
            if any(np.linalg.norm(point - q) < 0.9 for q in chosen):
                continue
            if chosen:
                try:
                    self.path(chosen[0], point)
                except ValueError:
                    continue
            chosen.append(point)
            if len(chosen) == count:
                return np.array(chosen)
        raise ValueError("Not enough connected inspection sites")
