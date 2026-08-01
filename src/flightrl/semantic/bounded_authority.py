from __future__ import annotations

from dataclasses import asdict, replace
from math import hypot, isfinite
from time import time
from typing import Any

from flightrl.hardware.aideck_stream import AiDeckFrame

from .contract import GroundingResult
from .controller import DiscoveryCommand, DiscoveryPhase


class PufferBoundedAuthority:
    """Apply short, forward-only Puffer control after an evidence-bound gate."""

    approved_authority = "bounded_forward_yaw"

    def __init__(self, policy, readiness: dict[str, Any]) -> None:
        self.policy = policy
        limits = readiness["limits"]
        self.forward_limit = float(limits["max_forward_speed_m_s"])
        self.search_yaw_limit = float(limits["search_abs_yawrate_deg_s"])
        self.detected_yaw_limit = float(limits["detected_abs_yawrate_deg_s"])
        self.stale_s = float(limits["proposal_stale_s"])
        self.duration_s = float(limits["max_authority_duration_s"])
        self.displacement_m = float(limits["max_displacement_m"])
        self.clearance_m = float(limits["minimum_predicted_clearance_m"])
        self.collision_risk = float(limits["maximum_predicted_collision_risk"])
        self._started_s: float | None = None
        self._origin_xy: tuple[float, float] | None = None
        self._latest_update_s: float | None = None
        self._latest_forward = 0.0
        self._latest_yawrate = 0.0
        self._latest_proposal: dict[str, Any] = {}

    def update(
        self,
        frame: AiDeckFrame,
        grounding: GroundingResult,
        telemetry: dict[str, float],
        *,
        now_s: float | None = None,
    ) -> dict[str, Any]:
        current_time = time() if now_s is None else now_s
        detection = None if grounding.best is None else asdict(grounding.best)
        proposal = self.policy.step(
            frame=frame.pixels,
            telemetry=telemetry,
            prompt=grounding.prompt,
            detection=detection,
        )
        forward = float(proposal["vx_body_m_s"])
        yawrate = float(proposal["yawrate_deg_s"])
        clearance = float(proposal["predicted_clearance_m"])
        collision_risk = float(proposal["predicted_collision_risk"])
        if not all(isfinite(value) for value in (forward, yawrate, clearance, collision_risk)):
            raise RuntimeError("Puffer bounded-control proposal is not finite")
        if self._started_s is None:
            self._started_s = current_time
            self._origin_xy = _position_xy(telemetry)
        budget_exhausted = self._budget_exhausted(current_time, telemetry)
        safety_stop = clearance < self.clearance_m or collision_risk > self.collision_risk
        self._latest_forward = (
            0.0
            if budget_exhausted or safety_stop
            else max(0.0, min(self.forward_limit, forward))
        )
        yaw_limit = (
            self.detected_yaw_limit if detection is not None else self.search_yaw_limit
        )
        self._latest_yawrate = (
            0.0
            if budget_exhausted
            else max(-yaw_limit, min(yaw_limit, yawrate))
        )
        self._latest_update_s = current_time
        self._latest_proposal = {
            **proposal,
            "monitor_only": False,
            "controls_drone": True,
            "approved_authority": self.approved_authority,
            "safety_stop": safety_stop,
            "authority_budget_exhausted": budget_exhausted,
            "applied_vx_body_m_s": self._latest_forward,
            "applied_vy_body_m_s": 0.0,
            "applied_vz_m_s": 0.0,
            "applied_yawrate_deg_s": self._latest_yawrate,
        }
        return self._latest_proposal.copy()

    def apply(
        self,
        baseline: DiscoveryCommand,
        *,
        now_s: float | None = None,
    ) -> DiscoveryCommand:
        current_time = time() if now_s is None else now_s
        terminal = baseline.phase in {DiscoveryPhase.COMPLETE, DiscoveryPhase.TIMEOUT}
        stale = (
            self._latest_update_s is None
            or current_time - self._latest_update_s > self.stale_s
        )
        return replace(
            baseline,
            vx_body_m_s=0.0 if terminal or stale else self._latest_forward,
            vy_body_m_s=0.0,
            yawrate_deg_s=0.0 if terminal or stale else self._latest_yawrate,
        )

    @property
    def latest_proposal(self) -> dict[str, Any]:
        return self._latest_proposal.copy()

    def _budget_exhausted(
        self,
        now_s: float,
        telemetry: dict[str, float],
    ) -> bool:
        assert self._started_s is not None
        assert self._origin_xy is not None
        position = _position_xy(telemetry)
        displacement = hypot(
            position[0] - self._origin_xy[0],
            position[1] - self._origin_xy[1],
        )
        return now_s - self._started_s >= self.duration_s or displacement >= self.displacement_m


def _position_xy(telemetry: dict[str, float]) -> tuple[float, float]:
    return (
        float(telemetry.get("stateEstimate.x", 0.0)),
        float(telemetry.get("stateEstimate.y", 0.0)),
    )
