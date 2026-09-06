"""Camera-derived navigation features and a shared, explicit clearance supervisor."""

import cv2
import numpy as np
from .sensing import DETECTOR, servo


def inspect_marker(rgb, depth, target_id):
    corners, ids, _ = DETECTOR.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    if ids is None:
        return None
    for points, ident in zip(corners, ids.flatten()):
        if int(ident) != target_id:
            continue
        p = points[0]
        center = p.mean(0)
        mask = np.zeros(depth.shape, np.uint8)
        cv2.fillConvexPoly(mask, (center + (p - center) * 0.6).astype(np.int32), 1)
        distances = depth[(mask > 0) & np.isfinite(depth)]
        if len(distances) < 8:
            return None
        f = rgb.shape[0] / (2 * np.tan(np.deg2rad(63) / 2))
        u = (center[0] - (rgb.shape[1] - 1) / 2) / f
        v = (center[1] - (rgb.shape[0] - 1) / 2) / f
        forward = float(np.median(distances)) / np.sqrt(1 + u * u + v * v)
        transform = cv2.getPerspectiveTransform(
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32), p.astype(np.float32)
        )
        uv = cv2.perspectiveTransform(
            np.array([[[0.5, -0.4333]]], np.float32), transform
        )[0, 0]
        x, y = np.rint(uv).astype(int)
        signal = None
        if 3 <= x < rgb.shape[1] - 3 and 3 <= y < rgb.shape[0] - 3:
            hsv = cv2.cvtColor(rgb[y - 2 : y + 3, x - 2 : x + 3], cv2.COLOR_RGB2HSV)
            h, s, b = np.median(hsv.reshape(-1, 3), axis=0)
            if s > 90 and b > 35:
                if h < 15 or h > 165:
                    signal = 1
                elif 30 < h < 90:
                    signal = 0
        ys, xs = np.where((mask > 0) & np.isfinite(depth))
        take = np.linspace(0, len(xs) - 1, min(400, len(xs))).astype(int)
        rays = np.c_[
            np.ones(len(take)),
            -(xs[take] - (rgb.shape[1] - 1) / 2) / f,
            -(ys[take] - (rgb.shape[0] - 1) / 2) / f,
        ]
        points = (
            rays
            / np.linalg.norm(rays, axis=1, keepdims=True)
            * depth[ys[take], xs[take], None]
        )
        covariance = np.cov(points.T)
        normal = np.linalg.eigh(covariance)[1][:, 0]
        if normal[0] > 0:
            normal = -normal
        return dict(
            normal=normal.astype(np.float32),
            relative=np.array([forward, -u * forward, -v * forward], np.float32),
            signal=signal,
            pixels=float(cv2.contourArea(p)),
        )
    return None


def clearance(depth):
    h, w = depth.shape
    sectors = []
    for left, right in ((0.1, 0.35), (0.35, 0.65), (0.65, 0.9)):
        values = depth[int(h * 0.43) : int(h * 0.57), int(w * left) : int(w * right)]
        values = values[np.isfinite(values)]
        sectors.append(float(np.quantile(values, 0.08)) if len(values) > 8 else 0.0)
    return np.array(sectors, np.float32)


def features(measurement, depth, proprio, kind):
    relative = (
        np.zeros(3, np.float32) if measurement is None else measurement["relative"]
    )
    return np.r_[
        relative / [8, 4, 4],
        float(measurement is not None),
        clearance(depth) / 8,
        [-1, 0, 0] if measurement is None else measurement["normal"],
        np.asarray(proprio, np.float32),
        [1, 0] if kind == "drone" else [0, 1],
    ].astype(np.float32)


def supervise(command, depth, kind, valid=True):
    command = np.asarray(command, np.float32).copy()
    if not valid:
        return np.zeros(4, np.float32)
    distance = clearance(depth)[1]
    stop = 0.7 if kind == "drone" else 0.42
    command[0] = min(command[0], max(0, (distance - stop) * 0.8))
    if kind == "rover":
        command[1:3] = 0
    return command


def teacher(measurement, depth, kind):
    if measurement is None:
        command, found = servo(None, kind)
        if kind == "rover":
            command[3] = 0.5
    else:
        relative = measurement["relative"]
        normal = measurement["normal"]
        error = relative + normal * (1.05 if kind == "drone" else 0.55)
        yaw = np.arctan2(-normal[1], -normal[0])
        command = np.array(
            [
                np.clip(error[0] * 0.6, -0.15, 0.4),
                np.clip(error[1] * 0.8, -0.35, 0.35),
                np.clip(error[2] * 0.8, -0.35, 0.35),
                np.clip(yaw * 1.2, -0.6, 0.6),
            ],
            np.float32,
        )
        if kind == "rover":
            command[1:3] = 0
            command[3] = np.clip(np.arctan2(relative[1], relative[0]) * 1.4, -0.6, 0.6)
        found = (
            abs(error[0]) < 0.15
            and abs(error[1]) < 0.12
            and (kind == "rover" or (abs(error[2]) < 0.12 and abs(yaw) < 0.18))
        )
    return supervise(command, depth, kind), found
