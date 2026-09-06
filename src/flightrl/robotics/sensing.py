"""Timestamped robot observations and image-only fiducial perception."""

from dataclasses import dataclass
import cv2
import numpy as np
from flightrl.fleet.camera_policy.sensors import proprioception

SIZES = ((512, 384), (128, 96))
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
DETECTOR = cv2.aruco.ArucoDetector(DICTIONARY)


def decode(data, count=2):
    if len(data) != count * sum(w * h * 8 for w, h in SIZES):
        raise ValueError("Incomplete camera batch")
    result = []
    offset = 0
    for _ in range(count):
        levels = []
        for w, h in SIZES:
            n = w * h * 4
            rgb = (
                np.frombuffer(data, np.uint8, count=n, offset=offset)
                .reshape(h, w, 4)[..., :3]
                .copy()
            )
            offset += n
            depth = (
                np.frombuffer(data, "<f4", count=w * h, offset=offset)
                .reshape(h, w)
                .copy()
            )
            offset += n
            if (
                not np.isfinite(depth).all()
                or (depth <= 0).any()
                or (depth > 8.001).any()
            ):
                raise ValueError("Invalid metric depth")
            levels.append((rgb, depth))
        result.append(levels)
    return result


@dataclass(frozen=True)
class Observation:
    rgb: np.ndarray
    depth: np.ndarray
    proprio: np.ndarray
    sequence: int
    time_s: float


def body_sensors(world, body):
    velocity = np.zeros(6)
    import mujoco as mj

    mj.mj_objectVelocity(
        world.model, world.data, mj.mjtObj.mjOBJ_BODY, body, velocity, 0
    )
    q = world.data.xquat[body][None].astype(np.float32)
    rates = world.data.xmat[body].reshape(3, 3).T @ velocity[:3]
    return proprioception(
        velocity[None, 3:].astype(np.float32), q, rates[None].astype(np.float32)
    )[0]


def target_from_pixels(rgb, depth, target_id):
    corners, ids, _ = DETECTOR.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    if ids is None:
        return None
    for c, ident in zip(corners, ids.flatten()):
        if int(ident) != target_id:
            continue
        points = c[0]
        center = points.mean(0)
        polygon = np.zeros(depth.shape, np.uint8)
        inset = center + (points - center) * 0.6
        cv2.fillConvexPoly(polygon, inset.astype(np.int32), 1)
        distances = depth[polygon > 0]
        if not len(distances):
            return None
        f = rgb.shape[0] / (2 * np.tan(np.deg2rad(63) * 0.5))
        u = (center[0] - (rgb.shape[1] - 1) * 0.5) / f
        v = (center[1] - (rgb.shape[0] - 1) * 0.5) / f
        forward = float(np.median(distances)) / np.sqrt(1 + u * u + v * v)
        return np.array([forward, -u * forward, -v * forward], np.float32)
    return None


def servo(relative, kind):
    if relative is None:
        return np.array([0, 0, 0, 0.22], np.float32), False
    x, y, z = relative
    stand = 1.05 if kind == "drone" else 0.55
    command = np.array(
        [
            np.clip((x - stand) * 0.6, -0.15, 0.4),
            np.clip(y * 0.8, -0.35, 0.35),
            np.clip(z * 0.8, -0.35, 0.35),
            np.clip(y * 0.3, -0.3, 0.3),
        ],
        np.float32,
    )
    if kind == "rover":
        command[1:3] = 0
        command[3] = np.clip(np.arctan2(y, x) * 1.4, -0.6, 0.6)
    found = (
        abs(x - stand) < 0.15 and abs(y) < 0.12 and (kind == "rover" or abs(z) < 0.12)
    )
    return command, found
