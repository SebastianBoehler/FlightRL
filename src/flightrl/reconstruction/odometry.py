"""Incremental RGB-D PnP and monocular triangulation baselines, no supplied poses."""

import cv2
import numpy as np
from .geometry import unproject


class VisualOdometry:
    def __init__(self, k, mode):
        if mode not in ("rgb", "rgbd"):
            raise ValueError("Unsupported reconstruction mode")
        self.k = k
        self.mode = mode
        self.pose = np.eye(4)
        self.previous = None
        self.pixels = None
        self.landmarks = None
        self.depth = None
        self.initialized = False
        self.status = "empty"
        self.inliers = 0
        cv2.setNumThreads(2)
        cv2.setRNGSeed(17)

    def features(self, gray):
        corners = cv2.goodFeaturesToTrack(gray, 600, 0.01, 7, blockSize=5)
        return np.empty((0, 2), np.float32) if corners is None else corners[:, 0]

    def step(self, rgb, depth=None):
        if self.mode == "rgbd" and depth is None:
            raise ValueError("RGB-D requires measured depth")
        if self.mode == "rgb" and depth is not None:
            raise ValueError("Monocular backend must not receive depth")
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if self.previous is None:
            self.previous = gray
            self.pixels = self.features(gray)
            self.depth = depth.copy() if depth is not None else None
            self.status = "initializing" if self.mode == "rgb" else "tracking"
            return self.pose.copy() if self.mode == "rgbd" else None
        if len(self.pixels) < 8:
            self.status = "lost"
            return None
        tracked, valid, _ = cv2.calcOpticalFlowPyrLK(
            self.previous, gray, self.pixels, None, winSize=(21, 21), maxLevel=3
        )
        if tracked is None or valid is None:
            self.status = "lost"
            return None
        reverse, back, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.previous, tracked, None, winSize=(21, 21), maxLevel=3
        )
        if reverse is None or back is None:
            self.status = "lost"
            return None
        good = (
            (valid[:, 0] > 0)
            & (back[:, 0] > 0)
            & (np.linalg.norm(reverse - self.pixels, axis=1) < 1)
        )
        h, w = gray.shape
        good &= (
            (tracked[:, 0] >= 0)
            & (tracked[:, 0] < w)
            & (tracked[:, 1] >= 0)
            & (tracked[:, 1] < h)
        )
        source = self.pixels[good]
        target = tracked[good]
        if len(source) < 8:
            self.status = "lost"
            return None
        if self.mode == "rgb" and not self.initialized:
            if np.median(np.linalg.norm(source - target, axis=1)) < 2:
                self.status = "initializing"
                return None
            e, mask = cv2.findEssentialMat(
                source, target, self.k, method=cv2.RANSAC, prob=0.999, threshold=1.0
            )
            if e is None or e.shape != (3, 3):
                self.status = "lost"
                return None
            n, r, t, mask = cv2.recoverPose(e, source, target, self.k, mask=mask)
            if n < 12:
                self.status = "lost"
                return None
            a = self.k @ np.eye(3, 4)
            b = self.k @ np.c_[r, t]
            points = cv2.triangulatePoints(a, b, source.T, target.T)
            points = (points[:3] / points[3]).T
            keep = (
                (mask[:, 0] > 0)
                & np.isfinite(points).all(1)
                & (points[:, 2] > 0)
                & (points[:, 2] < 100)
            )
            if keep.sum() < 12:
                self.status = "lost"
                return None
            self.pose[:3, :3] = r.T
            self.pose[:3, 3] = (-r.T @ t).ravel()
            self.landmarks = points[keep]
            self.pixels = target[keep]
            self.previous = gray
            self.initialized = True
            self.status = "tracking"
            self.inliers = int(keep.sum())
            return self.pose.copy()
        if self.mode == "rgbd":
            xy = np.clip(np.rint(source).astype(int), [0, 0], [w - 1, h - 1])
            dist = self.depth[xy[:, 1], xy[:, 0]]
            keep = np.isfinite(dist) & (dist > 0.1) & (dist < 7.95)
            objects = unproject(source[keep], dist[keep], self.k)
            image = target[keep]
        else:
            objects = self.landmarks[good]
            image = target
        if len(objects) < 8:
            self.status = "lost"
            return None
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objects,
            image,
            self.k,
            None,
            iterationsCount=100,
            reprojectionError=2.0,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or inliers is None or len(inliers) < 8:
            self.status = "lost"
            return None
        selected = inliers[:, 0]
        rvec, tvec = cv2.solvePnPRefineLM(
            objects[selected], image[selected], self.k, None, rvec, tvec
        )
        r, _ = cv2.Rodrigues(rvec)
        relative = np.eye(4)
        relative[:3, :3] = r.T
        relative[:3, 3] = (-r.T @ tvec).ravel()
        candidate = self.pose @ relative if self.mode == "rgbd" else relative
        if (
            np.linalg.norm(candidate[:3, 3] - self.pose[:3, 3]) > 1.5
            and self.mode == "rgbd"
        ):
            self.status = "lost"
            return None
        self.pose = candidate
        self.inliers = len(inliers)
        self.status = "tracking"
        self.previous = gray
        if self.mode == "rgbd":
            self.pixels = self.features(gray)
            self.depth = depth.copy()
        else:
            self.pixels = image[inliers[:, 0]]
            self.landmarks = objects[inliers[:, 0]]
        return self.pose.copy()
