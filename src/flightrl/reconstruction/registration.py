"""Overlap-verified RGB-D submap alignment; reject insufficient correspondence."""

import cv2
import numpy as np
from .geometry import unproject


def register(anchor_rgb, anchor_depth, other_rgb, k):
    orb = cv2.ORB_create(nfeatures=1500, edgeThreshold=10, fastThreshold=8)
    ka, da = orb.detectAndCompute(anchor_rgb, None)
    kb, db = orb.detectAndCompute(other_rgb, None)
    if da is None or db is None:
        return None, 0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    matches = [
        a
        for pair in pairs
        if len(pair) == 2
        for a, b in [pair]
        if a.distance < 0.7 * b.distance
    ]
    if len(matches) < 12:
        return None, len(matches)
    a = np.array([ka[m.queryIdx].pt for m in matches])
    b = np.array([kb[m.trainIdx].pt for m in matches])
    xy = np.clip(
        np.rint(a).astype(int),
        [0, 0],
        [anchor_depth.shape[1] - 1, anchor_depth.shape[0] - 1],
    )
    depth = anchor_depth[xy[:, 1], xy[:, 0]]
    valid = (depth > 0.1) & (depth < 7.95)
    if valid.sum() < 12:
        return None, int(valid.sum())
    xyz = unproject(a[valid], depth[valid], k)
    ok, rv, tv, inliers = cv2.solvePnPRansac(
        xyz,
        b[valid],
        k,
        None,
        reprojectionError=1.5,
        iterationsCount=200,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < 12:
        return None, 0
    r, _ = cv2.Rodrigues(rv)
    pose = np.eye(4)
    pose[:3, :3] = r.T
    pose[:3, 3] = (-r.T @ tv).ravel()
    return pose, len(inliers)
