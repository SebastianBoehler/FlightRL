from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import cos, radians, sin

from .contract import GroundingDetection, GroundingResult


class DiscoveryPhase(str, Enum):
    SCAN = "scan"
    REACQUIRE = "reacquire"
    TRACK = "track"
    HOLD = "hold"
    REPOSITION = "reposition"
    COMPLETE = "complete"
    TIMEOUT = "timeout"


@dataclass(frozen=True, slots=True)
class DiscoveryConfig:
    minimum_confidence: float = 0.25
    grounding_stale_s: float = 2.5
    search_yawrate_deg_s: float = 20.0
    track_yawrate_deg_s: float = 12.0
    yaw_gain: float = 30.0
    center_deadband: float = 0.08
    centered_hold_s: float = 1.5
    minimum_scan_s: float = 0.0
    reacquire_yawrate_deg_s: float = 12.0
    reacquire_tolerance_deg: float = 8.0
    max_duration_s: float = 45.0
    search_radius_m: float = 0.25
    reposition_speed_m_s: float = 0.06
    waypoint_tolerance_m: float = 0.07
    allow_reposition: bool = False
    image_error_to_yaw_sign: float = -1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum confidence must be in [0, 1]")
        positive = (
            self.grounding_stale_s,
            self.search_yawrate_deg_s,
            self.track_yawrate_deg_s,
            self.yaw_gain,
            self.center_deadband,
            self.centered_hold_s,
            self.reacquire_yawrate_deg_s,
            self.reacquire_tolerance_deg,
            self.max_duration_s,
            self.reposition_speed_m_s,
            self.waypoint_tolerance_m,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("discovery timing, gains, and rates must be positive")
        if self.minimum_scan_s < 0.0 or self.minimum_scan_s >= self.max_duration_s:
            raise ValueError("minimum scan time must be non-negative and below max duration")
        if self.search_radius_m < 0.0:
            raise ValueError("search radius must be non-negative")
        if self.allow_reposition and self.search_radius_m <= self.waypoint_tolerance_m:
            raise ValueError("search radius must exceed waypoint tolerance")
        if self.image_error_to_yaw_sign not in (-1.0, 1.0):
            raise ValueError("image-to-yaw sign must be -1 or 1")


@dataclass(frozen=True, slots=True)
class DiscoveryCommand:
    phase: DiscoveryPhase
    vx_body_m_s: float
    vy_body_m_s: float
    yawrate_deg_s: float
    target_visible: bool
    target_confidence: float
    horizontal_error: float
    waypoint_index: int


class DiscoveryController:
    def __init__(self, config: DiscoveryConfig, *, start_time_s: float) -> None:
        self.config = config
        self.start_time_s = start_time_s
        self.phase = DiscoveryPhase.SCAN
        self.phase_started_s = start_time_s
        self.centered_since_s: float | None = None
        self.waypoint_index = 0
        self.initial_scan_complete = config.minimum_scan_s == 0.0
        self.best_scan_confidence = 0.0
        self.reacquire_yaw_deg: float | None = None

    def step(
        self,
        *,
        now_s: float,
        grounding: GroundingResult | None,
        position_xy_m: tuple[float, float],
        origin_xy_m: tuple[float, float],
        yaw_deg: float,
    ) -> DiscoveryCommand:
        if now_s - self.start_time_s >= self.config.max_duration_s:
            self.phase = DiscoveryPhase.TIMEOUT
            return self._stationary(False, 0.0, 0.0)

        if not self.initial_scan_complete:
            detection = self._valid_detection(grounding, now_s)
            if detection is not None and detection.confidence > self.best_scan_confidence:
                self.best_scan_confidence = detection.confidence
                self.reacquire_yaw_deg = yaw_deg
            if now_s - self.start_time_s < self.config.minimum_scan_s:
                self.phase = DiscoveryPhase.SCAN
                return self._scan()
            self.initial_scan_complete = True
            self.phase = DiscoveryPhase.SCAN
            self.phase_started_s = now_s

        if self.reacquire_yaw_deg is not None:
            yaw_error = _signed_angle_deg(self.reacquire_yaw_deg - yaw_deg)
            if abs(yaw_error) > self.config.reacquire_tolerance_deg:
                self.phase = DiscoveryPhase.REACQUIRE
                yawrate = max(
                    -self.config.reacquire_yawrate_deg_s,
                    min(self.config.reacquire_yawrate_deg_s, yaw_error),
                )
                return DiscoveryCommand(
                    DiscoveryPhase.REACQUIRE,
                    0.0,
                    0.0,
                    yawrate,
                    False,
                    self.best_scan_confidence,
                    0.0,
                    self.waypoint_index,
                )
            self.reacquire_yaw_deg = None
            self.phase = DiscoveryPhase.SCAN
            self.phase_started_s = now_s

        detection = self._valid_detection(grounding, now_s)
        if detection is not None:
            return self._track(detection, now_s)

        self.centered_since_s = None
        if self.phase is not DiscoveryPhase.REPOSITION:
            self._set_phase(DiscoveryPhase.SCAN, now_s, preserve_scan_start=True)
        scan_duration_s = 360.0 / self.config.search_yawrate_deg_s
        if now_s - self.phase_started_s < scan_duration_s:
            return self._scan()
        if not self.config.allow_reposition:
            self.phase = DiscoveryPhase.TIMEOUT
            return self._stationary(False, 0.0, 0.0)
        return self._reposition(now_s, position_xy_m, origin_xy_m, yaw_deg)

    def _track(self, detection: GroundingDetection, now_s: float) -> DiscoveryCommand:
        error = detection.box.center_x - 0.5
        if abs(error) <= self.config.center_deadband:
            if self.centered_since_s is None:
                self.centered_since_s = now_s
            elapsed = now_s - self.centered_since_s
            self.phase = (
                DiscoveryPhase.COMPLETE
                if elapsed >= self.config.centered_hold_s
                else DiscoveryPhase.HOLD
            )
            return self._stationary(True, detection.confidence, error)

        self.centered_since_s = None
        self._set_phase(DiscoveryPhase.TRACK, now_s)
        yawrate = self.config.image_error_to_yaw_sign * self.config.yaw_gain * error
        yawrate = max(-self.config.track_yawrate_deg_s, min(self.config.track_yawrate_deg_s, yawrate))
        return DiscoveryCommand(
            DiscoveryPhase.TRACK,
            0.0,
            0.0,
            yawrate,
            True,
            detection.confidence,
            error,
            self.waypoint_index,
        )

    def _reposition(
        self,
        now_s: float,
        position_xy_m: tuple[float, float],
        origin_xy_m: tuple[float, float],
        yaw_deg: float,
    ) -> DiscoveryCommand:
        waypoints = self._waypoints(origin_xy_m)
        if self.waypoint_index >= len(waypoints):
            self.phase = DiscoveryPhase.TIMEOUT
            return self._stationary(False, 0.0, 0.0)
        target_x, target_y = waypoints[self.waypoint_index]
        dx = target_x - position_xy_m[0]
        dy = target_y - position_xy_m[1]
        distance = (dx * dx + dy * dy) ** 0.5
        if distance <= self.config.waypoint_tolerance_m:
            self.waypoint_index += 1
            self._set_phase(DiscoveryPhase.SCAN, now_s)
            return self._stationary(False, 0.0, 0.0)

        self._set_phase(DiscoveryPhase.REPOSITION, now_s, preserve_scan_start=True)
        speed = min(self.config.reposition_speed_m_s, distance)
        vx_world = speed * dx / distance
        vy_world = speed * dy / distance
        yaw = radians(yaw_deg)
        vx_body = cos(yaw) * vx_world + sin(yaw) * vy_world
        vy_body = -sin(yaw) * vx_world + cos(yaw) * vy_world
        return DiscoveryCommand(
            DiscoveryPhase.REPOSITION,
            vx_body,
            vy_body,
            0.0,
            False,
            0.0,
            0.0,
            self.waypoint_index,
        )

    def _valid_detection(
        self,
        grounding: GroundingResult | None,
        now_s: float,
    ) -> GroundingDetection | None:
        if grounding is None or now_s - grounding.frame_host_time_s > self.config.grounding_stale_s:
            return None
        best = grounding.best
        if best is None or best.confidence < self.config.minimum_confidence:
            return None
        return best

    def _waypoints(self, origin: tuple[float, float]) -> tuple[tuple[float, float], ...]:
        x, y = origin
        radius = self.config.search_radius_m
        return ((x + radius, y), (x, y + radius), (x - radius, y), (x, y - radius))

    def _stationary(
        self,
        visible: bool,
        confidence: float,
        error: float,
    ) -> DiscoveryCommand:
        return DiscoveryCommand(
            self.phase,
            0.0,
            0.0,
            0.0,
            visible,
            confidence,
            error,
            self.waypoint_index,
        )

    def _scan(self) -> DiscoveryCommand:
        return DiscoveryCommand(
            DiscoveryPhase.SCAN,
            0.0,
            0.0,
            self.config.search_yawrate_deg_s,
            False,
            0.0,
            0.0,
            self.waypoint_index,
        )

    def _set_phase(
        self,
        phase: DiscoveryPhase,
        now_s: float,
        *,
        preserve_scan_start: bool = False,
    ) -> None:
        if self.phase is phase:
            return
        previous = self.phase
        self.phase = phase
        if not (preserve_scan_start and previous is DiscoveryPhase.SCAN):
            self.phase_started_s = now_s


def _signed_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0
