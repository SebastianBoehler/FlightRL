"""Decode completed WebGPU frames without changing the frozen actor camera contract."""

import numpy as np
from flightrl.fleet.camera_policy.sensors import CameraPacket, proprioception

SIZES = ((256, 192), (64, 48))
FRAME_BYTES = 3 * sum(w * h * 8 for w, h in SIZES)


def decode_frames(data):
    if len(data) != FRAME_BYTES:
        raise ValueError(f"Incomplete RGB-D payload: {len(data)} != {FRAME_BYTES}")
    result = []
    offset = 0
    for _ in range(3):
        levels = []
        for width, height in SIZES:
            size = width * height * 4
            rgb = (
                np.frombuffer(data, np.uint8, count=size, offset=offset)
                .reshape(height, width, 4)[..., :3]
                .copy()
            )
            offset += size
            depth = (
                np.frombuffer(data, "<f4", count=width * height, offset=offset)
                .reshape(height, width)
                .copy()
            )
            offset += size
            if (
                not np.isfinite(depth).all()
                or (depth <= 0).any()
                or (depth > 8.001).any()
            ):
                raise ValueError(
                    "Camera must return finite metric ray distances in (0,8]"
                )
            levels.append((rgb, depth))
        result.append(levels)
    return result


def actor_packet(frames, state, messages):
    q = np.asarray(state["quaternions"][:3], np.float32)[:, [3, 0, 1, 2]]
    return CameraPacket(
        np.stack([x[1][0] for x in frames]),
        np.stack([x[1][1] for x in frames]),
        proprioception(
            np.asarray(state["velocities"], np.float32),
            q,
            np.asarray(state["rates"], np.float32),
        ),
        np.eye(3, dtype=np.float32),
        messages,
        state["sequence"],
        state["time_s"],
    )
