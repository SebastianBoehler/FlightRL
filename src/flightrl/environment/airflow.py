"""Vectorized reduced-order airflow. No pressure solve or CFD claim."""

import numpy as np
from flightrl.sixdof.geometry import quat_to_matrix


class Airflow:
    def __init__(self, profile, room, boxes, rng):
        self.profile, self.room, self.boxes, self.rng = profile, room, boxes, rng
        self.gust = np.zeros(3)
        self.time = 0.0

    def advance(self, dt):
        decay = np.exp(-dt / self.profile.correlation_s)
        self.gust = decay * self.gust + np.sqrt(1 - decay**2) * self.rng.normal(
            0, self.profile.turbulence_m_s, 3
        )
        self.time += dt

    def sample(self, points, drone, quaternion, thrust_ratio, *, wake=True,
               rotor_centers=((-0.049, -0.049, 0), (-0.049, 0.049, 0),
                              (0.049, -0.049, 0), (0.049, 0.049, 0)),
               rotor_radius=.055):
        points = np.asarray(points).reshape(-1, 3)
        # Spatially coherent gust variation; the same field is sampled by body and dust.
        modulation = 0.8 + 0.2 * np.sin(
            points[:, 0] * 1.7 + points[:, 1] * 0.9 - self.time
        )
        flow = (
            np.array(self.profile.wind_m_s)[None, :] + modulation[:, None] * self.gust
        )
        if wake:
            rotation = quat_to_matrix(np.asarray(quaternion)[None])[0]
            axis = rotation[:, 2]
            for center in rotor_centers:
                rotor = np.asarray(drone) + rotation @ np.asarray(center)
                delta = points - rotor
                below = -delta @ axis
                radial = delta + below[:, None] * axis
                radius = rotor_radius + 0.16 * np.maximum(below, 0)
                jet = np.exp(-np.sum(radial**2, axis=1) / (2 * radius**2)) * np.exp(
                    -np.maximum(below, 0) / 2
                )
                jet *= (
                    np.exp(np.minimum(below, 0) / 0.035)
                    * self.profile.downwash_m_s
                    * np.sqrt(max(0, thrust_ratio))
                    * 0.25
                )
                flow -= jet[:, None] * axis
                # Ground impingement turns the wake into a radial wall jet.
                floor = np.exp(-np.maximum(points[:, 2] - self.room[4], 0) / 0.15)
                direction = radial / np.maximum(
                    np.linalg.norm(radial, axis=1, keepdims=True), 0.02
                )
                flow += (
                    direction * (jet * floor * 2)[:, None]
                    + axis * (jet * floor * 1.2)[:, None]
                )
            # Reduced-order impinging wake: radial ground jet and rolling return flow.
            # Strength decays with rotor height and vanishes with thrust.
            offset = points - np.asarray(drone)
            radial_xy = offset[:, :2]
            distance = np.linalg.norm(radial_xy, axis=1)
            height = np.maximum(points[:, 2] - self.room[4], 0)
            rotor_height = max(float(drone[2]) - self.room[4], 0.1)
            spread = 4 * rotor_radius + 0.32 * rotor_height
            ring = np.exp(-(((distance - spread) / (spread * 0.65)) ** 2))
            envelope = np.exp(-((height / (0.45 + 0.35 * rotor_height)) ** 2))
            strength = self.profile.downwash_m_s * np.sqrt(max(0, thrust_ratio))
            strength *= np.exp(-rotor_height / 2)
            phase = distance * 8 - self.time * 4 + offset[:, 0] * 2
            # Keep the empirical floor return below the rotor disk; otherwise
            # its upward flow meets a discontinuous jet and traps grains there.
            envelope *= 1 - np.exp(-((np.maximum(rotor_height - height, 0) / 0.2) ** 2))
            lift = ring * envelope * strength * (0.65 + 0.25 * np.sin(phase))
            flow[:, 2] += lift
            direction = radial_xy / np.maximum(distance[:, None], 0.02)
            flow[:, :2] += direction * (lift * (1 - height / spread))[:, None]
            swirl = np.stack((-direction[:, 1], direction[:, 0]), axis=1)
            flow[:, :2] += swirl * (lift * 0.35 * np.sin(phase))[:, None]
        # Impermeable authored boxes: remove inward velocity near their surfaces.
        for box in self.boxes:
            for k in range(3):
                other = [j for j in range(3) if j != k]
                spans = np.all(
                    (points[:, other] >= np.asarray(box)[2 * np.array(other)] - 0.02)
                    & (
                        points[:, other]
                        <= np.asarray(box)[2 * np.array(other) + 1] + 0.02
                    ),
                    axis=1,
                )
                for side in (0, 1):
                    distance = points[:, k] - box[2 * k + side]
                    near = spans & (abs(distance) < 0.10)
                    # Only obstacle exterior; room response is applied below.
                    inward = flow[:, k] * (1 if side == 0 else -1) > 0
                    exterior = distance * (1 if side else -1) >= 0
                    mask = near & inward & exterior
                    flow[mask, k] *= np.clip(abs(distance[mask]) / 0.10, 0, 1)
        for k in range(3):
            for side in (0, 1):
                distance = abs(points[:, k] - self.room[2 * k + side])
                inward = flow[:, k] * (-1 if side == 0 else 1) > 0
                flow[inward, k] *= np.clip(distance[inward] / 0.1, 0, 1)
        return flow
