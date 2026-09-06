"""Native actuator dynamics, Jolt solid contacts, fixed 20 ms physics steps."""

from .jolt import NativeWorld
import numpy as np
from flightrl.fleet.vehicles import VEHICLES
from flightrl.fleet.camera_policy.sensors import rotations
from flightrl.sixdof.native import native_step
from .scene import decode_scene


class ContactWorld:
    dt = 0.02

    def __init__(self, payload):
        vertices, indices, self.scene_hash = decode_scene(payload)
        self.world = NativeWorld()
        self.solid = self.world.mesh(vertices, indices)
        self.specs = payload["bodies"]
        self.handles = [self.world.box(b) for b in self.specs]
        self.world.step(0)
        self.params = np.stack([VEHICLES[b["vehicle"]].physics() for b in self.specs[:3]])
        self.ambient_wind = np.array(payload["wind_m_s"], float)
        self.wind = np.tile(self.ambient_wind, (3, 1))
        self.thrust = np.ones(3, np.float32)
        self.actions = np.zeros((3, 4), np.float32)
        self.ticks = 0
        self.contacts = []
        self.total_contacts = 0
        self.sync()

    def sync(self):
        states = [self.world.get_body_stats(h) for h in self.handles]
        self.positions = np.array([s[0] for s in states])
        self.quaternions = np.array([s[1] for s in states])
        self.p = np.ascontiguousarray(self.positions[:3], np.float32)
        self.q = np.ascontiguousarray(self.quaternions[:3, [3, 0, 1, 2]], np.float32)
        self.v = np.ascontiguousarray([s[2] for s in states[:3]], np.float32)
        omega = np.array([self.world.get_angular_velocity(h) for h in self.handles[:3]])
        self.rates = np.ascontiguousarray(
            np.einsum("nji,nj->ni", rotations(self.q), omega), np.float32
        )

    def step(self):
        p, v, q, rates = (a.copy() for a in (self.p, self.v, self.q, self.rates))
        native_step(
            p,
            v,
            q,
            rates,
            np.empty((3, 6), np.float32),
            self.actions,
            self.dt,
            np.array([-40, 40, -40, 40, -5, 40, 8], np.float32),
            self.thrust,
            self.params,
        )
        # Jolt integrates gravity once. Native prediction supplies actuator acceleration.
        acceleration = (v - self.v) / self.dt
        acceleration[:, 2] += 9.81
        acceleration += self.params[:, 2, None] * self.wind
        omega = np.einsum("nij,nj->ni", rotations(self.q), rates)
        for i, handle in enumerate(self.handles[:3]):
            self.world.apply_force(
                handle, *map(float, acceleration[i] * self.params[i, 0])
            )
            self.world.set_angular_velocity(handle, *map(float, omega[i]))
        self.world.step(self.dt)
        self.contacts = self.world.contacts()
        self.total_contacts += len(self.contacts)
        self.ticks += 1
        self.sync()

    def rays(self, starts, directions, distance=1.0):
        return self.world.rays(starts, directions, distance)

    def drop_props(self):
        for handle, spec in zip(self.handles[3:], self.specs[3:]):
            self.world.set_transform(
                handle, tuple(spec["position"]), tuple(spec["quaternion"])
            )
            self.world.set_linear_velocity(handle, 0, 0, 0)
            self.world.set_angular_velocity(handle, 0.5, 0.2, 0.1)
        self.world.step(0)
        self.sync()

    def state(self, mode):
        return {
            "sequence": self.ticks,
            "time_s": round(self.ticks * self.dt, 8),
            "positions": self.positions.tolist(),
            "quaternions": self.quaternions.tolist(),
            "velocities": self.v.tolist(),
            "rates": self.rates.tolist(),
            "mode": mode,
            "contacts": self.total_contacts,
            "wind_m_s": self.wind.mean(axis=0).tolist(),
        }
