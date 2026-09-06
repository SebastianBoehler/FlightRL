"""Sensor-only actor boundary. Simulator geometry and positions never enter packets."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CameraPacket:
    rgb: np.ndarray
    depth: np.ndarray
    proprio: np.ndarray  # body velocity estimate, body gravity direction, gyro
    role: np.ndarray  # three role flags
    messages: np.ndarray  # two detected bits, two validity bits, two message ages
    sequence: int
    capture_time_s: float

    def validate(self):
        n = len(self.rgb)
        expected = [
            (self.rgb, (n, 48, 64, 3), np.uint8),
            (self.depth, (n, 48, 64), np.float32),
            (self.proprio, (n, 9), np.float32),
            (self.role, (n, 3), np.float32),
            (self.messages, (n, 6), np.float32),
        ]
        for value, shape, dtype in expected:
            if (
                value.shape != shape
                or value.dtype != dtype
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"Invalid sensor tensor; expected {shape} {dtype}")
        if not np.all((self.role == 0) | (self.role == 1)) or not np.all(
            self.role.sum(1) == 1
        ):
            raise ValueError("Role must be one-hot")
        if (
            not np.all((self.messages[:, :4] == 0) | (self.messages[:, :4] == 1))
            or (self.messages[:, 4:] < 0).any()
        ):
            raise ValueError("Peer reports require explicit bits and nonnegative ages")
        if (
            (self.depth < 0).any()
            or self.sequence < 0
            or not np.isfinite(self.capture_time_s)
            or self.capture_time_s < 0
        ):
            raise ValueError("Invalid depth or sensor timestamp")


def rotations(q):
    w, x, y, z = q.T
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(-1, 3, 3)


def proprioception(v, q, rates):
    r = rotations(q)
    body_v = np.einsum("nji,nj->ni", r, v)
    gravity = r[:, 2, :]
    return np.ascontiguousarray(np.c_[body_v, gravity, rates], np.float32)
