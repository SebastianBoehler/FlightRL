"""Explicit scene demonstration controller; uses simulator position, not a learned actor."""

import numpy as np
from flightrl.fleet.camera_policy.sensors import rotations


class DemoFlight:
    def __init__(self, world):
        self.world = world
        self.home = np.array([s["position"] for s in world.specs[:3]], float)
        self.started = world.ticks * world.dt

    def controls(self, dust=False):
        world = self.world
        target = self.home.copy()
        if dust:
            # Smooth descent, then a small sweep through the finite dust bed.
            age = max(0, world.ticks * world.dt - self.started)
            blend = 0.5 - 0.5 * np.cos(min(age / 6, 1) * np.pi)
            clearance = np.array([max(.48, s["halfExtents"][2] + .60) if s["vehicle"] == "agriculture"
                                  else .48 for s in world.specs[:3]])
            target[:, 2] = self.home[:, 2] * (1 - blend) + clearance * blend
            target[:, 0] += 0.45 * np.sin(max(0, age - 6) * 0.35)
        accel = np.clip(2 * (target - world.p) - 2.3 * world.v, -2, 2)
        accel[:, 2] += 9.81
        desired_up = accel / np.linalg.norm(accel, axis=1)[:, None]
        rotation = rotations(world.q)
        current_up = rotation[:, :, 2]
        omega = 5 * np.cross(current_up, desired_up)
        # Keep the original heading toward the beacons.
        omega[:, 2] -= 2 * np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])
        body_rates = np.einsum("nji,nj->ni", rotation, omega)
        thrust = np.sum(accel * current_up, axis=1) / 9.81
        action = np.c_[
            (thrust - 1) / world.params[:, 4], body_rates / world.params[:, 5:8]
        ]
        return np.ascontiguousarray(np.clip(action, -1, 1), np.float32)
