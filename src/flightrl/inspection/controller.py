"""Classical mission planner with observation-only panel memory and route recovery."""

import numpy as np
from flightrl.inspection_mission import detect_markers
from flightrl.inspection.mapping import ObservedMap, FY
from flightrl.sixdof.geometry import quat_to_matrix


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class MissionController:
    scan_ticks = 65
    waypoint_tolerance = 0.22

    def __init__(self, start):
        self.map = ObservedMap()
        self.home = np.array(start[:2])
        self.panels = {}
        self.inspected = set()
        self.breadcrumbs = [self.home.copy()]
        self.return_route = []
        self.mode = "scan"
        self.scan_left = self.scan_ticks
        self.route = []
        self.target = None
        self.tick = 0
        self.previous_link = True
        self.events = []
        self.hold = 0
        self.finished = False
        self.recovered = False
        self.known_done_scans = 0

    def observe(self, rgb, depth, position, quaternion, connected):
        self.tick += 1
        self.map.update(depth, position, quaternion)
        rotation = quat_to_matrix(quaternion[None])[0]
        origin = position + rotation @ np.array([0.035, 0, 0.012])
        for detection in detect_markers(rgb):
            key = detection["marker"]
            x0, y0, x1, y1 = detection["bbox_xyxy"]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            ray = np.array(
                [
                    1,
                    -(2 * (cx + 0.5) / 64 - 1) * FY * 64 / 48,
                    -(2 * (cy + 0.5) / 48 - 1) * FY,
                ]
            )
            ray /= np.linalg.norm(ray)
            point = origin + rotation @ ray * depth[int(cy), int(cx)]
            # Fit the visible panel plane from RGB-D pixels, orient toward camera.
            cols = np.arange(x0, x1 + 1)
            row = int(cy)
            rays = np.stack(
                (
                    np.ones_like(cols),
                    -(2 * (cols + 0.5) / 64 - 1) * FY * 64 / 48,
                    np.full_like(cols, -(2 * (row + 0.5) / 48 - 1) * FY, dtype=float),
                ),
                axis=1,
            )
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)
            points = origin + (rays * depth[row, cols, None]) @ rotation.T
            tangent = points[-1, :2] - points[0, :2]
            normal = np.array([-tangent[1], tangent[0]])
            normal /= max(np.linalg.norm(normal), 1e-6)
            if np.dot(normal, position[:2] - point[:2]) < 0:
                normal = -normal
            if key not in self.panels:
                self.events.append(
                    {"tick": self.tick - 1, "type": "discovered", "marker": key}
                )
            self.panels[key] = (point[:2], point[:2] + normal * 1.5)
            if detection["useful_view_observed"] and key not in self.inspected:
                # Conservative optical depth check in addition to pixel quality.
                if depth[int(cy), int(cx)] < 2.8:
                    self.inspected.add(key)
                    self.events.append(
                        {"tick": self.tick - 1, "type": "inspected", "marker": key}
                    )
                    if key == self.target:
                        self.route = []
                        self.target = None
        if (
            connected
            and self.mode != "recover"
            and np.linalg.norm(position[:2] - self.breadcrumbs[-1]) > 0.4
        ):
            self.breadcrumbs.append(position[:2].copy())
        if not connected and self.previous_link:
            self.mode = "recover"
            self.return_route = list(reversed(self.breadcrumbs))
            self.route = []
            self.events.append({"tick": self.tick - 1, "type": "link_lost"})
        if connected and not self.previous_link:
            self.events.append({"tick": self.tick - 1, "type": "reconnected"})
            if self.mode == "recover":
                self.recovered = True
                self.finished = True
                self.mode = "reconnected"
        self.previous_link = connected

    def command(self, position, quaternion):
        yaw = 2 * np.arctan2(quaternion[3], quaternion[0])
        goal = position[:2]
        face = yaw
        if self.finished:
            return np.zeros(4, np.float32), np.zeros(4, np.float32)
        if self.mode == "recover":
            while (
                self.return_route
                and np.linalg.norm(position[:2] - self.return_route[0]) < 0.3
            ):
                self.return_route.pop(0)
            if not self.return_route:
                self.mode = "outage_at_start"
                self.finished = True
                self.events.append({"tick": self.tick - 1, "type": "outage_at_start"})
                return np.zeros(4, np.float32), np.zeros(4, np.float32)
            goal = self.return_route[0]
            face = np.arctan2(*(goal - position[:2])[::-1])
            cell = self.map.cell(goal)
            if cell not in self.map.safe():
                self.mode = "recovery_blocked"
                self.finished = True
                self.events.append({"tick": self.tick - 1, "type": "recovery_blocked"})
                return np.zeros(4, np.float32), np.zeros(4, np.float32)
        elif self.scan_left > 0:
            self.mode = "scan"
            self.scan_left -= 1
            return np.array(
                [0, 0, np.clip((1.5 - position[2]) * 2, -1, 1), 0.15], np.float32
            ), np.array([0, 0, 0, 0.15], np.float32)
        else:
            if not self.route:
                targets = sorted(
                    set(self.panels) - self.inspected,
                    key=lambda k: np.linalg.norm(self.panels[k][1] - position[:2]),
                )
                for key in targets:
                    path = self.map.path(position, self.panels[key][1])
                    if path:
                        self.route = path[1:]
                        self.target = key
                        break
                if not self.route:
                    for cell in self.map.frontier(position)[:20]:
                        path = self.map.path(position, self.map.center(cell))
                        if len(path) > 1:
                            self.route = path[1:]
                            self.target = None
                            break
                if not self.route:
                    self.scan_left = self.scan_ticks
                    self.known_done_scans += 1
                    if self.known_done_scans >= 3:
                        self.finished = True
                        self.mode = "exploration_exhausted"
                    return np.zeros(4, np.float32), np.zeros(4, np.float32)
            while (
                self.route
                and np.linalg.norm(self.route[0] - position[:2])
                < self.waypoint_tolerance
            ):
                self.route.pop(0)
            if not self.route:
                if self.target:
                    point = self.panels[self.target][0]
                    face = np.arctan2(*(point - position[:2])[::-1])
                    self.hold += 1
                    if self.hold > 40:
                        self.target = None
                        self.hold = 0
                        self.scan_left = self.scan_ticks
                else:
                    self.scan_left = self.scan_ticks
            else:
                goal = self.route[0]
                face = np.arctan2(*(goal - position[:2])[::-1])
                self.hold = 0
            self.mode = "inspect" if self.target else "explore"
        delta = goal - position[:2]
        c, s = np.cos(yaw), np.sin(yaw)
        body = np.array([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]])
        desired = np.zeros(4, np.float32)
        desired[:2] = np.clip(body * 1.8, -1, 1)
        desired[2] = np.clip((1.5 - position[2]) * 2, -1, 1)
        desired[3] = np.clip(wrap(face - yaw) * 0.3, -0.2, 0.2)
        if abs(wrap(face - yaw)) > 0.5:
            desired[:2] = 0
        return desired, np.array([*body, 1.5 - position[2], desired[3]], np.float32)
