from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from flightrl.puffer4_config import Puffer4ExportSettings
from flightrl.puffer4_door_challenge_specs import resolve_door_challenge
from flightrl.puffer4_door_runtime import DoorMissionPhase
from flightrl.puffer4_door_sections import build_fixed_door_sections
from flightrl.semantic.contract import GroundingDetection, NormalizedBox


def _contract_module():
    spec = importlib.util.find_spec(
        "flightrl.puffer4_door_evidence_age_contract"
    )
    assert spec is not None, "fixed-door evidence-age contract is not implemented"
    return importlib.import_module(
        "flightrl.puffer4_door_evidence_age_contract"
    )


def test_approved_evidence_age_contract_drives_runtime_consumers() -> None:
    contracts = _contract_module()
    contract = contracts.FIXED_DOOR_EVIDENCE_AGE_CONTRACT
    sections = build_fixed_door_sections(Puffer4ExportSettings())
    phase = DoorMissionPhase()
    _, transform, _ = resolve_door_challenge(
        "camera-latency-92ms",
        {
            "control_dt": 1.0 / 65.0,
            "maximum_evidence_age_s": 1.0,
            "camera_mean_min": 18.0,
            "camera_mean_max": 110.0,
            "camera_randomization": 0.0,
            "obstacle_probability": 0.0,
            "layout_diversity": 1.0,
            "room_x_min": -2.0,
            "room_x_max": 2.0,
            "room_y_min": -2.0,
            "room_y_max": 2.0,
        },
        agent_count=4,
    )

    assert sections["env"]["control_dt"] == pytest.approx(1.0 / 65.0)
    assert sections["env"]["maximum_evidence_age_s"] == 1.0
    contract.verify_env(sections["env"])
    assert phase.maximum_detection_age_s == 1.0
    assert transform.mechanism_report()["control_dt_s"] == pytest.approx(
        1.0 / 65.0
    )
    assert transform.mechanism_report()["maximum_evidence_age_s"] == 1.0


def test_evidence_age_contract_rejects_rehashed_unapproved_runtime() -> None:
    contracts = _contract_module()
    report = contracts.FIXED_DOOR_EVIDENCE_AGE_CONTRACT.to_report()
    report["maximum_evidence_age_s"] = 1.1
    payload = {key: value for key, value in report.items() if key != "sha256"}
    report["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    with pytest.raises(ValueError, match="approved"):
        contracts.approved_door_evidence_age_contract_from_report(report)


def test_evidence_age_contract_fails_closed_on_environment_drift() -> None:
    contracts = _contract_module()
    contract = contracts.FIXED_DOOR_EVIDENCE_AGE_CONTRACT
    env = build_fixed_door_sections(Puffer4ExportSettings())["env"]
    env["maximum_evidence_age_s"] = 1.1

    with pytest.raises(ValueError, match="maximum_evidence_age_s"):
        contract.verify_env(env)


def test_camera_latency_challenge_rejects_evidence_age_drift() -> None:
    baseline = build_fixed_door_sections(Puffer4ExportSettings())["env"]
    baseline["obstacle_probability"] = 0.0
    baseline["layout_diversity"] = 1.0
    baseline["maximum_evidence_age_s"] = 1.1

    with pytest.raises(ValueError, match="maximum_evidence_age_s"):
        resolve_door_challenge(
            "camera-latency-92ms",
            baseline,
            agent_count=4,
        )


def test_host_clears_detection_at_approved_stale_boundary() -> None:
    detection = GroundingDetection(
        "door",
        0.9,
        NormalizedBox(0.1, 0.2, 0.3, 0.8),
    )

    phase = DoorMissionPhase().update(detection, age_s=1.0)

    assert phase.name == "search"
    assert phase.evidence == pytest.approx((0.0, 0.0, 0.0, 0.0, 1.0))


def test_shadow_detection_gate_uses_approved_stale_boundary() -> None:
    script = runpy.run_path(
        Path(__file__).parents[1] / "scripts/crazyflie_door_puffer_shadow.py",
        run_name="door_shadow_contract_test",
    )
    result = SimpleNamespace(
        frame_host_time_s=9.0,
        best=object(),
        inference_ms=2.0,
    )

    detection, age_s, _ = script["latest_detection"](
        (0, result),
        now_s=10.0,
    )

    assert detection is None
    assert age_s == 1.0


def test_shadow_detection_rejects_future_result_for_older_policy_frame() -> None:
    script = runpy.run_path(
        Path(__file__).parents[1] / "scripts/crazyflie_door_puffer_shadow.py",
        run_name="door_shadow_future_result_test",
    )
    result = SimpleNamespace(
        frame_index=12,
        frame_host_time_s=10.1,
        best=object(),
        inference_ms=2.0,
    )

    with pytest.raises(RuntimeError, match="newer than policy frame"):
        script["latest_detection"](
            (12, result),
            now_s=10.0,
            policy_frame_index=11,
        )
