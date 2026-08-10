from __future__ import annotations

from math import isfinite, pi

import numpy as np


class ScanAdvanceTeacher:
    """Privileged simulation teacher for an obvious advance-then-turn patrol."""

    privileged_teacher = True
    privileged_inputs = ("front_range_m",)
    flight_authority = False
    forward_action = 0.5
    leg_distance_m = 0.8
    turn_clearance_m = 0.65
    resume_clearance_m = 0.85
    turn_angle_rad = pi / 2.0

    def __init__(self) -> None:
        self.phase = "uninitialized"
        self.completed_turns = 0
        self._leg_origin = np.zeros(2, dtype=np.float32)
        self._last_yaw = 0.0
        self._scan_progress_rad = 0.0

    def reset(self, position_xy: np.ndarray, *, yaw_rad: float) -> None:
        self._leg_origin = _position(position_xy).copy()
        self._last_yaw = _yaw(yaw_rad)
        self._scan_progress_rad = 0.0
        self.completed_turns = 0
        self.phase = "advance"

    def action(
        self,
        position_xy: np.ndarray,
        *,
        yaw_rad: float,
        horizontal_ranges_m: np.ndarray,
    ) -> np.ndarray:
        if self.phase == "uninitialized":
            raise RuntimeError("scan-advance teacher must be reset before use")
        position = _position(position_xy)
        yaw = _yaw(yaw_rad)
        ranges = np.asarray(horizontal_ranges_m, dtype=np.float32)
        if ranges.shape != (4,) or not np.isfinite(ranges).all() or np.any(ranges <= 0.0):
            raise ValueError("teacher horizontal ranges must be four finite positive values")

        if self.phase == "advance":
            leg_distance = float(np.linalg.norm(position - self._leg_origin))
            if ranges[0] <= self.turn_clearance_m or leg_distance >= self.leg_distance_m:
                self.phase = "scan_turn"
                self._last_yaw = yaw
                self._scan_progress_rad = 0.0
                return self._turn_action()
            return self._forward_action()

        yaw_delta = float(
            np.arctan2(
                np.sin(yaw - self._last_yaw),
                np.cos(yaw - self._last_yaw),
            )
        )
        self._scan_progress_rad += max(0.0, yaw_delta)
        self._last_yaw = yaw
        if (
            self._scan_progress_rad + 1.0e-6 >= self.turn_angle_rad
            and ranges[0] >= self.resume_clearance_m
        ):
            self.completed_turns += 1
            self.phase = "advance"
            self._leg_origin = position.copy()
            return self._forward_action()
        return self._turn_action()

    def _forward_action(self) -> np.ndarray:
        return np.asarray((self.forward_action, 0.0, 0.0, 0.0), dtype=np.float32)

    def _turn_action(self) -> np.ndarray:
        return np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32)


def _position(value: np.ndarray) -> np.ndarray:
    position = np.asarray(value, dtype=np.float32)
    if position.shape != (2,) or not np.isfinite(position).all():
        raise ValueError("teacher position must contain finite XY")
    return position


def _yaw(value: float) -> float:
    if isinstance(value, bool) or not isfinite(float(value)):
        raise ValueError("teacher yaw must be finite")
    return float(value)
