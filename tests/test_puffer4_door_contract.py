from __future__ import annotations

import importlib
import importlib.util
import hashlib
import json

import pytest

from flightrl.puffer4_config import Puffer4ExportSettings
from flightrl.puffer4_door_sections import build_fixed_door_sections


def _contract_module():
    spec = importlib.util.find_spec("flightrl.puffer4_door_contract")
    assert spec is not None
    return importlib.import_module("flightrl.puffer4_door_contract")


def test_corrected_action_contract_is_hash_bound_and_drives_export() -> None:
    contracts = _contract_module()
    contract = contracts.CORRECTED_DOOR_ACTION_CONTRACT
    report = contract.to_report()

    restored = contracts.DoorActionContract.from_report(report)
    sections = build_fixed_door_sections(Puffer4ExportSettings())

    assert restored == contract
    assert sections["env"]["max_horizontal_speed_m_s"] == 0.55
    assert sections["env"]["max_yawrate_deg_s"] == 70.0
    assert sections["env"]["max_rate_yaw"] == 4.0


def test_action_contract_rejects_report_mutation() -> None:
    contracts = _contract_module()
    report = contracts.CORRECTED_DOOR_ACTION_CONTRACT.to_report()
    report["max_yawrate_deg_s"] = 229.1831180523293

    with pytest.raises(ValueError, match="SHA-256"):
        contracts.DoorActionContract.from_report(report)


@pytest.mark.parametrize(
    ("contract_id", "yawrate"),
    (
        ("fixed-door-declared-yaw-v1", 71.0),
        ("fixed-door-unreviewed-v1", 70.0),
    ),
)
def test_action_contract_registry_rejects_self_consistent_unapproved_contract(
    contract_id: str,
    yawrate: float,
) -> None:
    contracts = _contract_module()
    report = contracts.CORRECTED_DOOR_ACTION_CONTRACT.to_report()
    report["contract_id"] = contract_id
    report["max_yawrate_deg_s"] = yawrate
    payload = {key: value for key, value in report.items() if key != "sha256"}
    report["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    with pytest.raises(ValueError, match="approved"):
        contracts.approved_door_action_contract_from_report(report)


def test_action_contract_applies_and_verifies_native_environment_values() -> None:
    contracts = _contract_module()
    contract = contracts.CORRECTED_DOOR_ACTION_CONTRACT
    env = {
        "max_horizontal_speed_m_s": 99.0,
        "max_yawrate_deg_s": 99.0,
        "max_rate_yaw": 99.0,
    }

    contract.apply_to_env(env)
    contract.verify_env(env)

    env["max_yawrate_deg_s"] = 229.1831180523293
    with pytest.raises(ValueError, match="max_yawrate_deg_s"):
        contract.verify_env(env)


def test_action_contract_rejects_unrealizable_or_false_legacy_yaw_mapping() -> None:
    contracts = _contract_module()
    common = {
        "contract_id": "test-only",
        "schema_version": 1,
        "max_forward_speed_m_s": 0.55,
        "physics_max_yawrate_rad_s": 4.0,
    }

    with pytest.raises(ValueError, match="exceeds physics yaw ceiling"):
        contracts.DoorActionContract(
            **common,
            max_yawrate_deg_s=230.0,
            native_yaw_mapping="declared_policy_rate",
        )
    with pytest.raises(ValueError, match="legacy yaw mapping"):
        contracts.DoorActionContract(
            **common,
            max_yawrate_deg_s=70.0,
            native_yaw_mapping="legacy_direct_physics_ceiling",
        )


def test_v59_legacy_contract_is_explicit_not_a_default() -> None:
    contracts = _contract_module()
    legacy = contracts.LEGACY_V59_ACTION_CONTRACT

    assert legacy.contract_id == "fixed-door-v59-legacy-physics-yaw-v1"
    assert legacy.max_yawrate_deg_s == pytest.approx(229.1831180523293)
    assert legacy.native_yaw_action_scale == pytest.approx(1.0)
    assert legacy.native_yaw_mapping == "legacy_direct_physics_ceiling"


def test_live_safety_contract_is_separate_from_policy_scale() -> None:
    contracts = _contract_module()
    safety = contracts.FIXED_DOOR_LIVE_SAFETY_CONTRACT
    restored = contracts.DoorLiveSafetyContract.from_report(safety.to_report())

    assert restored == safety
    assert safety.contract_id == "fixed-door-yaw-only-live-v2"
    assert safety.max_yawrate_deg_s == 8.0
    assert safety.min_height_m == 0.20
    assert safety.max_height_m == 0.80
    assert safety.max_duration_s == 15.0
    assert safety.readiness_limits()["min_height_m"] == 0.20
    assert safety.readiness_limits()["max_height_m"] == 0.80
    assert safety.readiness_limits()["max_duration_s"] == 15.0
    assert safety.max_yawrate_deg_s < (
        contracts.CORRECTED_DOOR_ACTION_CONTRACT.max_yawrate_deg_s
    )
    assert safety.normalized_yaw_limit(
        contracts.CORRECTED_DOOR_ACTION_CONTRACT
    ) == pytest.approx(8.0 / 70.0)
    assert safety.translation_enabled is False

    mutated = safety.to_report()
    mutated["max_yawrate_deg_s"] = 9.0
    with pytest.raises(ValueError, match="SHA-256"):
        contracts.DoorLiveSafetyContract.from_report(mutated)


@pytest.mark.parametrize(
    ("height_m", "duration_s"),
    (
        (0.19, 15.0),
        (0.81, 15.0),
        (0.5, 15.01),
        (float("nan"), 15.0),
        (0.5, float("inf")),
    ),
)
def test_live_safety_contract_rejects_out_of_envelope_or_nonfinite_run(
    height_m: float,
    duration_s: float,
) -> None:
    contracts = _contract_module()

    with pytest.raises(ValueError, match="live (height|duration)"):
        contracts.FIXED_DOOR_LIVE_SAFETY_CONTRACT.require_live_envelope(
            height_m=height_m,
            duration_s=duration_s,
        )


@pytest.mark.parametrize("height_m", (0.20, 0.50, 0.80))
def test_live_safety_contract_accepts_reviewed_envelope_boundaries(
    height_m: float,
) -> None:
    contracts = _contract_module()

    contracts.FIXED_DOOR_LIVE_SAFETY_CONTRACT.require_live_envelope(
        height_m=height_m,
        duration_s=15.0,
    )
