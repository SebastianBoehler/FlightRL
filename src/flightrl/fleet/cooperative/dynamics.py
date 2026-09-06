"""Native six-DOF integration and swept collision checks, all positions in metres."""

import numpy as np
from flightrl import _binding
from flightrl.sixdof.native import native_step
from flightrl.fleet.vehicles import VEHICLES


class Flight:
    def __init__(self, scene, homes):
        self.p = np.array(homes, np.float32)
        self.v = np.zeros((3, 3), np.float32)
        self.q = np.zeros((3, 4), np.float32)
        self.q[:, 0] = 1
        self.rates = self.v.copy()
        self.ranges = np.empty((3, 6), np.float32)
        self.thrust = np.ones(3, np.float32)
        self.actions = np.zeros((3, 4), np.float32)
        self.physics = np.repeat(VEHICLES["fpv"].physics()[None], 3, axis=0)
        self.room = scene.scenario.arrays["terrain_bounds"]
        self.boxes = scene.scenario.arrays["terrain_obstacles"]
        self.radius = VEHICLES["fpv"].radius
        self.min_gap = float("inf")
        self.physics_steps = 0

    def step(self, waypoints):
        command = np.zeros((3, 4), np.float32)
        for i, target in enumerate(waypoints):
            delta = target - self.p[i]
            desired = np.clip(1.3 * delta - 0.15 * self.v[i], -0.6, 0.6)
            yaw = 2 * np.arctan2(self.q[i, 3], self.q[i, 0])
            c, s = np.cos(yaw), np.sin(yaw)
            command[i, :3] = [
                (c * desired[0] + s * desired[1]) / 0.7,
                (-s * desired[0] + c * desired[1]) / 0.7,
                desired[2] / 0.4,
            ]
            # Face travel direction so the re-rendered onboard views show progress.
            if np.linalg.norm(delta[:2]) > 0.3:
                bearing = np.arctan2(delta[1], delta[0])
                error = (bearing - yaw + np.pi) % (2 * np.pi) - np.pi
                command[i, 3] = np.clip(error * 0.6, -0.5, 0.5)
        _binding.sixdof_setpoint_actions(
            self.v, self.q, command, self.physics, self.actions, 0.7, 0.4, 2.5, 6.0, 3.0
        )
        return self.integrate()

    def integrate(self):
        """Advance supplied collective/body-rate actions without a setpoint controller."""
        for _ in range(5):
            before = self.p.copy()
            native_step(
                self.p,
                self.v,
                self.q,
                self.rates,
                self.ranges,
                self.actions,
                0.02,
                self.room,
                self.thrust,
                self.physics,
            )
            self.physics_steps += 1
            hit = np.zeros(3, np.uint8)
            _binding.inspection_collision(
                before, self.p, self.room, self.boxes, self.radius, hit
            )
            if hit.any():
                return True
            for i in range(3):
                for j in range(i):
                    a = before[i] - before[j]
                    d = self.p[i] - self.p[j] - a
                    t = np.clip(-np.dot(a, d) / max(np.dot(d, d), 1e-12), 0, 1)
                    gap = float(np.linalg.norm(a + t * d) - 2 * self.radius)
                    self.min_gap = min(self.min_gap, gap)
                    if gap <= 0:
                        return True
        return False
