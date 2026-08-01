from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from flightrl.puffer4_door_bundle import (
    FixedDoorCheckpointBundle,
    load_fixed_door_checkpoint_bundle,
)
from flightrl.puffer4_door_contract import DoorActionContract
from flightrl.puffer4_door_policy_contract import DoorPolicyArchitecture
from flightrl.puffer4_door_runtime import DoorPufferShadow
from flightrl.puffer4_door_snapshot import (
    FixedDoorCheckpointSnapshot,
    load_fixed_door_checkpoint_snapshot,
)
from flightrl.semantic.contract import GroundingDetection, NormalizedBox


class DoorPufferControlAdapter:
    """Expose the fixed-door student through the existing authority contract."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        action_contract: DoorActionContract,
        architecture: DoorPolicyArchitecture | None = None,
    ) -> None:
        self.runtime = DoorPufferShadow(
            checkpoint,
            architecture=architecture,
        )
        self.action_contract = action_contract
        self.bundle: FixedDoorCheckpointBundle | None = None
        self._executed_previous = np.zeros(2, dtype=np.float32)

    @classmethod
    def from_evaluation_report(
        cls,
        checkpoint: str | Path,
        report_path: str | Path,
    ) -> DoorPufferControlAdapter:
        bundle = load_fixed_door_checkpoint_bundle(checkpoint, report_path)
        adapter = cls(
            checkpoint,
            action_contract=bundle.action_contract,
            architecture=bundle.architecture,
        )
        adapter.bundle = bundle
        return adapter

    @classmethod
    def from_checkpoint_snapshot(
        cls,
        snapshot: FixedDoorCheckpointSnapshot,
        bundle: FixedDoorCheckpointBundle,
    ) -> DoorPufferControlAdapter:
        if (
            snapshot.source_path != bundle.checkpoint_path
            or snapshot.sha256 != bundle.checkpoint_sha256
        ):
            raise ValueError("fixed-door checkpoint snapshot identity changed")
        adapter = cls.__new__(cls)
        adapter.runtime = DoorPufferShadow.from_state_dict(
            snapshot.state_dict,
            architecture=bundle.architecture,
        )
        adapter.action_contract = bundle.action_contract
        adapter.bundle = bundle
        adapter._executed_previous = np.zeros(2, dtype=np.float32)
        return adapter

    def reset(self) -> None:
        self.runtime.reset()
        self._executed_previous.fill(0.0)

    def step(
        self,
        *,
        frame: np.ndarray,
        telemetry: dict[str, float],
        prompt: str,
        detection: dict[str, Any] | None,
        detection_age_s: float | None = 0.0,
    ) -> dict[str, float | bool | str]:
        if "door" not in prompt.strip().lower():
            raise ValueError("fixed-door policy only supports a door target")
        output = self.runtime.step(
            frame,
            telemetry,
            detection=_detection_from_dict(detection),
            detection_age_s=detection_age_s,
            executed_previous_action=self._executed_previous,
        )
        return {
            **output,
            "vx_body_m_s": (
                float(output["action_forward"])
                * self.action_contract.max_forward_speed_m_s
            ),
            "yawrate_deg_s": (
                float(output["action_yaw"])
                * self.action_contract.max_yawrate_deg_s
            ),
        }

    def record_executed_action(
        self,
        *,
        vx_body_m_s: float,
        yawrate_deg_s: float,
    ) -> None:
        self._executed_previous[:] = (
            np.clip(
                vx_body_m_s / self.action_contract.max_forward_speed_m_s,
                0.0,
                1.0,
            ),
            np.clip(
                yawrate_deg_s / self.action_contract.max_yawrate_deg_s,
                -1.0,
                1.0,
            ),
        )


def load_readiness_bound_control_adapter(
    checkpoint: str | Path,
    evaluation_report: str | Path,
    readiness: Mapping[str, Any],
) -> DoorPufferControlAdapter:
    """Snapshot policy after readiness and revalidate exact evidence."""
    bundle = require_readiness_bound_control_evidence(
        checkpoint,
        evaluation_report,
        readiness,
    )
    snapshot = load_fixed_door_checkpoint_snapshot(
        checkpoint,
        bundle.checkpoint_sha256,
    )
    refreshed = require_readiness_bound_control_evidence(
        checkpoint,
        evaluation_report,
        readiness,
    )
    return DoorPufferControlAdapter.from_checkpoint_snapshot(
        snapshot,
        refreshed,
    )


def require_readiness_bound_control_evidence(
    checkpoint: str | Path,
    evaluation_report: str | Path,
    readiness: Mapping[str, Any],
) -> FixedDoorCheckpointBundle:
    bundle = load_fixed_door_checkpoint_bundle(checkpoint, evaluation_report)
    expected = {
        "checkpoint path": (
            str(bundle.checkpoint_path),
            readiness.get("checkpoint"),
        ),
        "checkpoint SHA-256": (
            bundle.checkpoint_sha256,
            readiness.get("checkpoint_sha256"),
        ),
        "evaluation report path": (
            str(bundle.report_path),
            readiness.get("evaluation_report"),
        ),
        "evaluation report SHA-256": (
            bundle.report_sha256,
            readiness.get("evaluation_report_sha256"),
        ),
        "lineage report path": (
            str(bundle.lineage_report_path),
            readiness.get("lineage_report"),
        ),
        "lineage report SHA-256": (
            bundle.lineage_report_sha256,
            readiness.get("lineage_report_sha256"),
        ),
    }
    mismatched = [
        label
        for label, (actual, approved) in expected.items()
        if actual != approved
    ]
    if mismatched:
        raise ValueError(
            "fixed-door control evidence changed after readiness load: "
            + ", ".join(mismatched)
        )
    return bundle


def _detection_from_dict(
    detection: dict[str, Any] | None,
) -> GroundingDetection | None:
    if detection is None:
        return None
    box = detection["box"]
    return GroundingDetection(
        label=str(detection["label"]),
        confidence=float(detection["confidence"]),
        box=NormalizedBox(
            x_min=float(box["x_min"]),
            y_min=float(box["y_min"]),
            x_max=float(box["x_max"]),
            y_max=float(box["y_max"]),
        ),
        verification_confidence=_optional_float(
            detection.get("verification_confidence")
        ),
        verification_margin=_optional_float(detection.get("verification_margin")),
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
