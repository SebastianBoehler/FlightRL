"""Finite tracer population: inertial drag, settling, contact and resuspension."""

import numpy as np


class DustParticles:
    def __init__(self, profile, room, boxes, rng):
        self.profile, self.room, self.boxes = profile, room, boxes
        self.position = rng.uniform(
            room[:6:2] + 0.001, room[1:6:2] - 0.001, (profile.particle_count, 3)
        )
        bed_count = int(profile.particle_count * profile.settled_fraction)
        bed = np.asarray(
            profile.dust_bed_bounds if profile.dust_bed_bounds is not None else room[:4]
        )
        if np.any(bed[::2] < room[:4:2]) or np.any(bed[1::2] > room[1:4:2]):
            raise ValueError("dust bed outside room")
        self.position[:bed_count, :2] = rng.uniform(
            bed[::2] + 0.001, bed[1::2] - 0.001, (bed_count, 2)
        )
        self.position[:bed_count, 2] = room[4] + 0.001
        for _ in range(100):
            inside = np.zeros(len(self.position), bool)
            for b in boxes:
                inside |= np.all(
                    (self.position > b[::2]) & (self.position < b[1::2]), axis=1
                )
            if not inside.any():
                break
            self.position[inside] = rng.uniform(
                room[:6:2] + 0.001, room[1:6:2] - 0.001, (inside.sum(), 3)
            )
            bed_inside = np.flatnonzero(inside[:bed_count])
            self.position[bed_inside, :2] = rng.uniform(
                bed[::2] + 0.001, bed[1::2] - 0.001, (len(bed_inside), 2)
            )
            self.position[:bed_count, 2] = room[4] + 0.001
        else:
            raise ValueError("could not place dust outside solid geometry")
        self.diameter = (
            rng.uniform(*profile.grain_diameter_um, len(self.position)) * 1e-6
        )
        # Spherical mineral grains in room-temperature air: Stokes response time.
        self.relaxation_s = (
            profile.grain_density_kg_m3 * self.diameter**2 / (18 * 1.81e-5)
        )
        self.gravity = 9.80665 * (1 - 1.225 / profile.grain_density_kg_m3)
        self.velocity = np.zeros_like(self.position)
        self.active = np.ones(len(self.position), bool)
        self.active[:bed_count] = False
        self.deposited = 0
        self.resuspended = 0

    def step(self, flow, dt):
        speed = np.linalg.norm(flow, axis=1)
        lift = (
            (~self.active)
            & (speed > self.profile.resuspension_m_s)
            & (flow[:, 2] > 0)
            & (self.position[:, 2] < self.room[4] + 0.02)
        )
        self.active[lift] = True
        self.position[lift, 2] += 0.003
        self.resuspended += int(lift.sum())
        target = flow.copy()
        # Schiller-Naumann finite-Re correction; freeze drag over this substep.
        reynolds = (
            1.225
            * self.diameter
            * np.linalg.norm(flow - self.velocity, axis=1)
            / 1.81e-5
        )
        relaxation = self.relaxation_s / (1 + 0.15 * reynolds**0.687)
        target[:, 2] -= self.gravity * relaxation
        response = (1 - np.exp(-dt / relaxation))[:, None]
        self.velocity += (target - self.velocity) * response
        self.velocity[~self.active] = 0
        before = self.position.copy()
        # Persistent wall contact retains tangential motion rather than clipping
        # the entire displacement again on every subsequent step.
        for b in self.boxes:
            for k in range(3):
                other = [j for j in range(3) if j != k]
                spans = np.all(
                    (before[:, other] >= b[2 * np.array(other)])
                    & (before[:, other] <= b[2 * np.array(other) + 1]),
                    axis=1,
                )
                for side in (0, 1):
                    gap = (before[:, k] - b[2 * k + side]) * (1 if side else -1)
                    inward = self.velocity[:, k] * (1 if side == 0 else -1) > 0
                    self.velocity[spans & (gap >= 0) & (gap < 0.0001) & inward, k] = 0
        delta = self.velocity * dt
        after = before + delta
        contact = np.zeros(len(before), bool)
        # Swept slab intersections prevent particles teleporting through partitions.
        fraction = np.ones(len(before))
        normal_axis = np.full(len(before), -1)
        supported = np.zeros(len(before), bool)
        for b in self.boxes:
            lo = np.zeros(len(before))
            hi = np.ones(len(before))
            hit = np.ones(len(before), bool)
            entry_axis = np.zeros(len(before), int)
            for k in range(3):
                moving = abs(delta[:, k]) > 1e-12
                hit &= moving | (
                    (before[:, k] >= b[2 * k]) & (before[:, k] <= b[2 * k + 1])
                )
                a = np.full(len(before), -np.inf)
                z = np.full(len(before), np.inf)
                np.divide(b[2 * k] - before[:, k], delta[:, k], out=a, where=moving)
                np.divide(b[2 * k + 1] - before[:, k], delta[:, k], out=z, where=moving)
                entry = np.minimum(a, z)
                entry_axis[entry > lo] = k
                lo = np.maximum(lo, entry)
                hi = np.minimum(hi, np.maximum(a, z))
            hit &= (lo <= hi) & (lo >= 0) & (lo <= 1) & self.active
            closest = hit & (lo <= fraction)
            normal_axis[closest] = entry_axis[closest]
            supported[closest] = (entry_axis[closest] == 2) & (delta[closest, 2] < 0)
            fraction[hit] = np.minimum(fraction[hit], np.maximum(0, lo[hit] - 0.001))
            contact |= hit
        after = before + delta * fraction[:, None]
        escaped = np.any(
            (after <= self.room[:6:2] + 0.001) | (after >= self.room[1:6:2] - 0.001),
            axis=1,
        )
        contact |= escaped
        self.position[:] = np.clip(
            after, self.room[:6:2] + 0.001, self.room[1:6:2] - 0.001
        )
        # Walls remove normal motion; they do not glue grains in mid-air.
        for k in range(3):
            wall = (
                (normal_axis == k)
                | (after[:, k] <= self.room[2 * k] + 0.001)
                | (after[:, k] >= self.room[2 * k + 1] - 0.001)
            )
            self.velocity[wall, k] = 0
        supported |= after[:, 2] <= self.room[4] + 0.001
        self.deposited += int((supported & self.active).sum())
        self.active[supported] = False
        self.velocity[supported] = 0
