"""Seeded sensor errors and wheel odometry; ideal-state access is confined here."""

from collections import deque
import numpy as np
import mujoco as mj
from .sensing import body_sensors


def quaternion_from_euler(roll, pitch, yaw):
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


class SensorRig:
    def __init__(self, world, seed):
        self.rng = np.random.default_rng(seed + 18001)
        self.bias = self.rng.normal(0, 0.006, (2, 9))
        self.proprio = np.zeros((2, 9), np.float32)
        self.yaw = np.full(2, world.site["yaw"])
        self.position = np.array(world.site["spawns"], float)
        self.q = np.zeros((2, 4), np.float32)
        self.velocity = np.zeros((2, 3), np.float32)
        self.variance = np.full(2, 0.0001)
        self.elapsed = 0.0
        self.travelled = np.zeros(2)
        self.queue = deque()
        self.valid = [True, True]
        self.dropout_batches = 0
        self.last_capture = -1.0
        self.update(world, 0)

    def update(self, world, dt):
        self.elapsed += dt
        for i, body in enumerate((world.drone, world.rover)):
            measured = body_sensors(world, body).astype(float)
            if i == 1:
                wheel = []
                for name in ("left_wheel_joint", "right_wheel_joint"):
                    joint = mj.mj_name2id(world.model, mj.mjtObj.mjOBJ_JOINT, name)
                    wheel.append(world.data.qvel[world.model.jnt_dofadr[joint]])
                measured[0] = sum(wheel) * 0.11 / 2
                measured[1:3] = 0
            measured += self.bias[i] + self.rng.normal(0, 0.003, 9)
            self.proprio[i] = measured
            # Proprioception stores world-up in body axes and body angular rates.
            self.yaw[i] += measured[8] * dt
            g = measured[3:6]
            g = g / max(np.linalg.norm(g), 1e-6)
            roll = np.arctan2(g[1], g[2])
            pitch = np.arcsin(np.clip(-g[0], -1, 1))
            q = quaternion_from_euler(roll, pitch, self.yaw[i])
            r = np.zeros(9)
            mj.mju_quat2Mat(r, q)
            self.q[i] = q
            self.velocity[i] = r.reshape(3, 3) @ measured[:3]
            self.position[i] += self.velocity[i] * dt
            self.travelled[i] += np.linalg.norm(self.velocity[i]) * dt
            self.variance[i] = (
                0.0001
                + (0.006 * self.elapsed) ** 2
                + (0.006 * self.elapsed * self.travelled[i] / 2) ** 2
            )

    def deliver(self, frames, state, delivery_time_s=None):
        timestamp = state["time_s"]
        if timestamp <= self.last_capture:
            raise ValueError("Camera acquisition times must increase within an episode")
        self.last_capture = timestamp
        available = timestamp if delivery_time_s is None else delivery_time_s
        if available < timestamp:
            raise ValueError("Camera cannot arrive before acquisition")
        self.queue.append((frames, state))
        if available - self.queue[0][1]["time_s"] < 0.1 - 1e-9:
            return None
        # Deliver the newest eligible capture and explicitly count skipped observations.
        frames, state = self.queue.popleft()
        skipped = 0
        while self.queue and available - self.queue[0][1]["time_s"] >= 0.1 - 1e-9:
            frames, state = self.queue.popleft()
            skipped += 1
        state = {
            **state,
            "delivery_time_s": available,
            "observation_age_s": available - state["time_s"],
            "skipped_captures": skipped,
        }
        timestamp = state["time_s"]
        outage = 2.8 <= timestamp < 3.2 or 7 <= timestamp < 7.3
        self.valid = [not outage, not outage]
        if outage:
            self.dropout_batches += 1
        noisy = []
        for levels in frames:
            out = []
            for rgb, depth in levels:
                depth = depth + self.rng.normal(
                    0, 0.004 + 0.0008 * depth**2, depth.shape
                )
                depth = np.clip(depth, 0.01, 8).astype(np.float32)
                depth[self.rng.random(depth.shape) < 0.01] = np.nan
                if outage:
                    rgb = np.zeros_like(rgb)
                    depth[:] = np.nan
                out.append((rgb, depth))
            noisy.append(out)
        return noisy, state

    def record(self):
        return dict(
            velocity_source=["modeled velocity estimate", "wheel encoders"],
            pose_source="integrated biased velocity and gyro; known launch reference",
            camera_minimum_delay_s=0.1,
            camera_delivery="newest eligible capture; actual age recorded per observation",
            depth_noise="sigma=0.004+0.0008*range^2 m; 1% missing",
            dropout_intervals_s=[[2.8, 3.2], [7, 7.3]],
            dropout_batches=self.dropout_batches,
            estimated_positions_m=self.position.tolist(),
            position_variance_m2=self.variance.tolist(),
        )
