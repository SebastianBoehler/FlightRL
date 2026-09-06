"""Finite dust/leaves plus an explicit rain emitter, colliding with Jolt surfaces."""

import numpy as np
from flightrl.fleet.camera_policy.sensors import rotations
from flightrl.environment.airflow import Airflow
from flightrl.environment.profile import EnvironmentProfile
from flightrl.robotics.drone_asset import drone_model


class Particles:
    def __init__(self, world, seed=911):
        self.world = world
        self.models = [drone_model(s["vehicle"]) for s in world.specs[:3]]
        self.rng = np.random.default_rng(seed)
        self.kind = np.r_[np.zeros(1024, int), np.ones(160, int), np.full(320, 2, int)]
        self.p = self.rng.uniform([-5, -5, 0.2], [7, 5, 7], (len(self.kind), 3))
        self.v = np.zeros_like(self.p)
        self.support = np.full(len(self.p), -1, int)
        self.local = np.zeros_like(self.p)
        self.active = self.kind != 0
        dust = self.kind == 0
        # A finite dry bed around the three demonstration hover locations.
        ids = np.flatnonzero(dust)
        homes = np.array([s["position"] for s in world.specs[:3]])
        self.p[dust, :2] = homes[ids % 3, :2] + self.rng.uniform(
            -0.85, 0.85, (len(ids), 2)
        )
        extents = np.array([s["halfExtents"] for s in world.specs[:3]])
        self.p[dust, :2] = homes[ids % 3, :2] + (self.p[dust, :2] - homes[ids % 3, :2]) * np.maximum(1, extents[ids % 3, :2] * 2)
        starts = self.p[dust].copy()
        # Cast below the aircraft: its conservative rotor envelope is not a dust bed.
        starts[:, 2] = homes[ids % 3, 2] - extents[ids % 3, 2] - .1
        hits = world.rays(starts, np.tile([0, 0, -1], (dust.sum(), 1)), 12)
        if (hits["fraction"] >= 1).any():
            raise ValueError("Dust bed must have a solid support surface")
        self.p[dust] = (
            starts
            + np.array([0, 0, -12]) * hits["fraction"][:, None]
            + hits["normal"] * 0.002
        )
        self.anchor(np.flatnonzero(dust), hits["body"])
        # Effective leaf/rain terminal speeds; dust uses mineral-grain Stokes drag.
        diameter = self.rng.uniform(15, 40, len(self.kind)) * 1e-6
        self.tau = 2500 * diameter**2 / (18 * 1.81e-5)
        self.tau[self.kind == 1] = 0.45 / 9.81
        self.tau[self.kind == 2] = 6 / 9.81
        profile = EnvironmentProfile(
            name="shared-forest",
            surface_style="forest",
            wind_m_s=tuple(world.ambient_wind),
            downwash_m_s=2,
            turbulence_m_s=0.08,
        )
        self.air = Airflow(
            profile, np.array([-35, 35, -35, 35, -0.04, 40]), np.empty((0, 6)), self.rng
        )
        self.rain = False
        self.rain_emitted = 0
        self.rain_impacts = 0
        self.resuspended = 0
        self.escaped_dust = 0
        self.escaped = np.zeros(len(self.kind), bool)

    def anchor(self, ids, bodies):
        self.support[ids] = bodies
        for body in np.unique(bodies):
            if body <= 0:
                continue
            selected = ids[bodies == body]
            state = self.world.world.state(int(body))
            rotation = rotations(state[None, 3:7][:, [3, 0, 1, 2]])[0]
            self.local[selected] = (self.p[selected] - state[:3]) @ rotation

    def step(self):
        for body in np.unique(self.support[~self.active]):
            if body <= 0:
                continue
            selected = (self.support == body) & ~self.active & ~self.escaped
            state = self.world.world.state(int(body))
            rotation = rotations(state[None, 3:7][:, [3, 0, 1, 2]])[0]
            self.p[selected] = self.local[selected] @ rotation.T + state[:3]
        dt = self.world.dt
        self.air.advance(dt)
        self.world.wind = self.air.sample(
            self.world.p,
            self.world.p[0],
            self.world.q[0],
            self.world.thrust[0],
            wake=False,
        )
        flow = self.air.sample(
            self.p, self.world.p[0], self.world.q[0], self.world.thrust[0],
            rotor_centers=self.models[0]["rotor_centers_m"], rotor_radius=self.models[0]["rotor_radius_m"]
        )
        for i in (1, 2):
            flow += self.air.sample(
                self.p, self.world.p[i], self.world.q[i], self.world.thrust[i],
                rotor_centers=self.models[i]["rotor_centers_m"], rotor_radius=self.models[i]["rotor_radius_m"]
            ) - self.air.sample(
                self.p,
                self.world.p[i],
                self.world.q[i],
                self.world.thrust[i],
                wake=False,
            )
        dust = self.kind == 0
        lift = dust & ~self.active & ~self.escaped & (flow[:, 2] > 0.35)
        for x, y, z, speed in self.world.contacts:
            if speed > 0.3:
                nearby = (
                    dust
                    & ~self.active
                    & ~self.escaped
                    & (np.linalg.norm(self.p - [x, y, z], axis=1) < 0.45)
                )
                lift |= nearby
                self.v[nearby, 2] = min(1.5, speed * 0.25)
        self.support[lift] = -1
        self.active[lift] = True
        self.p[lift, 2] += 0.004
        self.resuspended += int(lift.sum())
        rain = self.kind == 2
        expired = rain & (self.p[:, 2] < -0.5)
        if self.rain:
            born = rain & (~self.active | expired)
            self.p[born] = self.rng.uniform([-5, -5, 6], [7, 5, 8], (born.sum(), 3))
            self.v[born] = [0.2, 0.08, -6]
            self.active[born] = True
            self.rain_emitted += int(born.sum())
        else:
            self.active[rain] = False
        target = flow.copy()
        target[:, 2] -= 9.81 * self.tau
        response = -np.expm1(-dt / self.tau)
        self.v += (target - self.v) * response[:, None]
        leaf = (self.kind == 1) & self.active
        self.v[leaf, 0] += 0.015 * np.sin(
            self.world.ticks * 0.13 + np.flatnonzero(leaf)
        )
        self.v[~self.active] = 0
        moving = np.flatnonzero(self.active)
        if len(moving):
            delta = self.v[moving] * dt
            hit = self.world.rays(self.p[moving], delta)
            fraction = hit["fraction"]
            collided = fraction < 1
            self.p[moving] += delta * fraction[:, None]
            ids = moving[collided]
            self.p[ids] += hit["normal"][collided] * 0.002
            self.active[ids] = False
            self.v[ids] = 0
            self.anchor(ids, hit["body"][collided])
            self.rain_impacts += int((self.kind[ids] == 2).sum())
        outside = (np.abs(self.p[:, :2]) > 35).any(1) | (self.p[:, 2] < -2)
        self.escaped_dust += int((outside & dust & ~self.escaped).sum())
        self.escaped |= outside & ~rain
        self.active[self.escaped] = False

    def record(self):
        visible = ~self.escaped & ((self.kind != 2) | self.active)
        return {
            "particles": self.p[visible].round(4).tolist(),
            "particleKinds": self.kind[visible].tolist(),
            "dust_airborne": int(((self.kind == 0) & self.active).sum()),
            "dust_settled": int(
                ((self.kind == 0) & ~self.active & ~self.escaped).sum()
            ),
            "dust_escaped": self.escaped_dust,
            "dust_resuspended": self.resuspended,
            "rain_emitted": self.rain_emitted,
            "rain_impacts": self.rain_impacts,
        }
