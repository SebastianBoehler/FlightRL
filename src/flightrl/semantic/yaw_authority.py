from __future__ import annotations

from dataclasses import asdict, replace
from math import isfinite
from time import time
from typing import Any

from flightrl.hardware.aideck_stream import AiDeckFrame

from .contract import GroundingResult
from .controller import DiscoveryCommand, DiscoveryPhase


class PufferYawAuthority:
    """Apply a gated Puffer proposal to yaw while denying all translation."""

    approved_authority = "yaw_only"

    def __init__(self, policy, readiness: dict[str, Any]) -> None:
        self.policy = policy
        limits = readiness["limits"]
        self.search_limit = float(limits["search_abs_yawrate_deg_s"])
        self.detected_limit = float(limits["detected_abs_yawrate_deg_s"])
        self.stale_s = float(limits["proposal_stale_s"])
        self._latest_yawrate = 0.0
        self._latest_update_s: float | None = None
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
            detection_age_s=max(
                0.0,
                current_time - grounding.frame_host_time_s,
            ),
        )
        proposed_yawrate = float(proposal["yawrate_deg_s"])
        if not isfinite(proposed_yawrate):
            raise RuntimeError("Puffer yaw proposal is not finite")
        limit = self.detected_limit if detection is not None else self.search_limit
        source_time = min(frame.host_time_s, grounding.frame_host_time_s)
        source_stale = current_time - source_time > self.stale_s
        self._latest_yawrate = (
            0.0
            if source_stale
            else max(-limit, min(limit, proposed_yawrate))
        )
        self._latest_update_s = source_time
        self._latest_proposal = {
            **proposal,
            "monitor_only": False,
            "controls_drone": True,
            "approved_authority": "yaw_only",
            "applied_vx_body_m_s": 0.0,
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
        terminal = baseline.phase in {
            DiscoveryPhase.COMPLETE,
            DiscoveryPhase.TIMEOUT,
        }
        stale = (
            self._latest_update_s is None
            or current_time - self._latest_update_s > self.stale_s
        )
        yawrate = 0.0 if terminal or stale else self._latest_yawrate
        record_executed = getattr(self.policy, "record_executed_action", None)
        if record_executed is not None:
            record_executed(
                vx_body_m_s=0.0,
                yawrate_deg_s=yawrate,
            )
        return replace(
            baseline,
            vx_body_m_s=0.0,
            vy_body_m_s=0.0,
            yawrate_deg_s=yawrate,
        )

    @property
    def latest_proposal(self) -> dict[str, Any]:
        return self._latest_proposal.copy()
