"""Seeded camera renders; no target coordinates are saved in actor input tensors."""

import numpy as np
from flightrl import _binding
from flightrl.environment import EnvironmentProfile

from .sensors import CameraPacket, proprioception
from .teacher import labels

APPEARANCE = EnvironmentProfile("camera_control").render_buffers()
COLORS = [(220, 30, 30), (30, 30, 220), (30, 220, 30)]
ROOM = np.array([-10, 10, -10, 10, 0, 10, 8], np.float32)


def dataset(seed, count):
    rng = np.random.default_rng(seed)
    rgb = np.empty((count, 48, 64, 3), np.uint8)
    depth = np.empty((count, 48, 64), np.float32)
    angles = rng.uniform(-0.12, 0.12, (count, 3))
    r, p, y = (angles / 2).T
    q = np.c_[
        np.cos(r) * np.cos(p) * np.cos(y) + np.sin(r) * np.sin(p) * np.sin(y),
        np.sin(r) * np.cos(p) * np.cos(y) - np.cos(r) * np.sin(p) * np.sin(y),
        np.cos(r) * np.sin(p) * np.cos(y) + np.sin(r) * np.cos(p) * np.sin(y),
        np.cos(r) * np.cos(p) * np.sin(y) - np.sin(r) * np.sin(p) * np.cos(y),
    ].astype(np.float32)
    v = rng.uniform(-0.5, 0.5, (count, 3)).astype(np.float32)
    role_ids = rng.integers(0, 3, count)
    roles = np.eye(3, dtype=np.float32)[role_ids]
    messages = np.c_[
        rng.integers(0, 2, (count, 2)),
        np.ones((count, 2)),
        rng.uniform(0, 0.8, (count, 2)),
    ].astype(np.float32)
    for i in range(count):
        distance = rng.uniform(0.75, 5)
        if i % 3 == 0:
            distance = rng.uniform(0.8, 1.4)
        lateral = rng.uniform(-0.5, 0.5) * distance
        vertical = rng.uniform(-0.32, 0.32) * distance
        panels = []
        for j, color in enumerate(COLORS):
            offset = 0 if j == role_ids[i] else (j - role_ids[i]) * 1.7
            panels.append(
                [
                    distance,
                    lateral + offset,
                    5 + vertical,
                    0,
                    -1,
                    0,
                    0,
                    0,
                    1,
                    0.3,
                    0.3,
                    *color,
                ]
            )
        boxes = np.empty((0, 6), np.float32)
        if i % 4 == 0:
            bx = rng.uniform(0.6, 2.5)
            by = rng.uniform(-1.5, 1.5)
            boxes = np.array(
                [[bx, bx + 0.2, by, by + 0.3, 0, rng.uniform(4.5, 5.5)]], np.float32
            )
        counts = np.empty((1, 3, 2), np.int32)
        _binding.inspection_render(
            np.array([[0, 0, 5]], np.float32),
            q[i : i + 1],
            ROOM,
            boxes,
            np.array(panels, np.float32),
            rgb[i : i + 1],
            counts,
            depth[i : i + 1],
            1,
            *APPEARANCE,
        )
    proprio = proprioception(
        v, q, rng.uniform(-0.4, 0.4, (count, 3)).astype(np.float32)
    )
    packet = CameraPacket(rgb, depth, proprio, roles, messages, 0, 0.0)
    action, found = labels(packet, q)
    return packet, action, found
