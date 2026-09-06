"""Own-engine coupling of ambient air, native dynamics, dust and sensor images."""

import numpy as np
from .airflow import Airflow
from .particles import DustParticles
from .aerosol import AerosolCamera
from .optics import CameraOptics


class EnvironmentSimulation:
    def __init__(self, seed, scene, sensor_size=(256, 192)):
        if scene.environment is None:
            raise ValueError("scene requires an environment profile")
        self.profile = scene.environment
        self.room = scene.scenario.arrays["terrain_bounds"]
        self.boxes = scene.scenario.arrays["terrain_obstacles"]
        self.air = Airflow(
            self.profile, self.room, self.boxes, np.random.default_rng(seed + 7100)
        )
        self.dust = DustParticles(
            self.profile, self.room, self.boxes, np.random.default_rng(seed + 8100)
        )
        self.aerosol = AerosolCamera(self.profile, self.room, self.boxes, *sensor_size)
        self.optics = CameraOptics(*sensor_size)
        self.gust = np.zeros(3)
        self.wind = np.zeros(3)
        self.render_buffers = self.profile.render_buffers()
        axes = [
            np.arange(self.room[2 * k] + 0.15, self.room[2 * k + 1], 0.75)
            for k in range(3)
        ]
        self.flow_points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
            -1, 3
        )
        for box in self.boxes:
            inside = np.all(
                (self.flow_points >= box[::2]) & (self.flow_points <= box[1::2]), axis=1
            )
            self.flow_points = self.flow_points[~inside]
        self.flow_samples = np.column_stack(
            (self.flow_points, np.zeros_like(self.flow_points))
        )

    def set_obstacles(self, boxes):
        self.boxes = boxes
        self.air.boxes = boxes
        self.dust.boxes = boxes
        self.aerosol.set_obstacles(boxes)

    @property
    def particles(self):
        return self.dust.position

    def step(self, velocity, dt, position, quaternion, thrust_ratio):
        if not np.isfinite(dt) or not 0 < dt <= 0.05:
            raise ValueError("environment dt must be in (0,.05] seconds")
        self.air.advance(dt)
        self.wind = self.air.sample(
            position, position, quaternion, thrust_ratio, wake=False
        )[0]
        # Native six-DOF supplies -drag*v. Adding drag*wind yields air-relative drag.
        # Own rotor wake is excluded from the body sample: thrust already models it.
        self.gust = self.profile.air_drag_per_s * self.wind
        velocity[0] += self.gust * dt
        flow = self.air.sample(self.dust.position, position, quaternion, thrust_ratio)
        self.dust.step(flow, dt)
        vectors = self.air.sample(self.flow_points, position, quaternion, thrust_ratio)
        self.flow_samples = np.column_stack((self.flow_points, vectors))

    def camera(self, rgb, depth, position, quaternion):
        self.aerosol.apply(rgb, depth, position, quaternion, self.dust)

    def record(self):
        return {
            "airflow_samples": getattr(self, "flow_samples", np.empty((0, 6)))
            .round(3)
            .tolist(),
            "gust_m_s2": self.gust.round(4).tolist(),
            "wind_m_s": self.wind.round(4).tolist(),
            "settled_particles": self.dust.position[~self.dust.active]
            .round(3)
            .tolist(),
            "particles": self.dust.position[self.dust.active].round(3).tolist(),
            "dust_airborne": int(self.dust.active.sum()),
            "dust_deposited": int((~self.dust.active).sum()),
            "dust_resuspensions": self.dust.resuspended,
            "mean_transmission": self.aerosol.mean_transmission,
        }
