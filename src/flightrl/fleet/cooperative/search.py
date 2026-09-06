"""Two scout roles and a confirmer; synthetic detection with geometric visibility."""

import numpy as np


def visible(a, b, boxes):
    """Segment/AABB occlusion, excluding an endpoint's supporting ground."""
    delta = np.asarray(b) - a
    for box in boxes:
        lower, upper = np.asarray(box)[::2], np.asarray(box)[1::2]
        enter, leave = 0.0, 1.0
        for axis in range(3):
            if abs(delta[axis]) < 1e-9:
                if a[axis] < lower[axis] or a[axis] > upper[axis]:
                    enter, leave = 1.0, 0.0
                    break
            else:
                t0, t1 = sorted(
                    (
                        (lower[axis] - a[axis]) / delta[axis],
                        (upper[axis] - a[axis]) / delta[axis],
                    )
                )
                enter, leave = max(enter, t0), min(leave, t1)
        if enter <= leave:
            return False
    return True


class SearchProtocol:
    roles = ["Scout A", "Scout B", "Confirmation drone"]

    def __init__(self, count):
        self.found = np.zeros(count, bool)
        self.reported_at = np.full(count, float("inf"))
        self.finders = np.full(count, -1)

    def eligible(self, drone, target, now):
        return not self.found[target] if drone < 2 else now >= self.reported_at[target]

    def inspect(self, drone, target, now, position, target_xy, boxes):
        if drone >= 2 and (not self.found[target] or now < self.reported_at[target]):
            raise ValueError("Cannot confirm before a scout report arrives")
        beacon = np.array([*target_xy, 0.65])
        if np.linalg.norm(position[:2] - target_xy) > 0.5 or not visible(
            position, beacon, boxes
        ):
            return None
        if drone < 2:
            self.found[target] = True
            self.finders[target] = drone
            self.reported_at[target] = now + 0.2
            return {
                "type": "detected",
                "text": f"Scout {drone + 1} found beacon {target + 1}; confirmation report sent",
            }
        return {
            "type": "confirmed",
            "text": f"Drone 3 confirmed beacon {target + 1} found by Scout {self.finders[target] + 1}",
        }
