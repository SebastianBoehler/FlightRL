from __future__ import annotations

import hashlib
import json

import pytest

from flightrl.puffer4_config import Puffer4ExportSettings
from flightrl.puffer4_door_contract import (
    DoorTeacherActionContract,
    PRIVILEGED_DOOR_TEACHER_ACTION_CONTRACT,
    door_teacher_action_contract_from_report,
)
from flightrl.puffer4_door_sections import build_fixed_door_teacher_sections
from flightrl.puffer4_edge_schema import ACTION_SPECS


def test_teacher_action_contract_is_hash_bound_and_drives_export() -> None:
    contract = PRIVILEGED_DOOR_TEACHER_ACTION_CONTRACT

    restored = DoorTeacherActionContract.from_report(contract.to_report())
    sections = build_fixed_door_teacher_sections(Puffer4ExportSettings())

    assert restored == contract
    assert contract.to_report()["privileged_tail_order"] == [
        "teacher_forward",
        "teacher_yaw",
        "visible",
        "center_x",
        "center_y",
        "scale",
    ]
    edge_scales = {name: scale for name, _unit, scale, _frame in ACTION_SPECS}
    assert sections["env"]["max_horizontal_speed_m_s"] == edge_scales["vx"]
    assert sections["env"]["max_vertical_speed_m_s"] <= edge_scales["vz"]
    assert sections["env"]["max_yawrate_deg_s"] == edge_scales["yaw_rate"]
    assert sections["env"]["max_rate_yaw"] == 4.0
    assert sections["train"]["total_timesteps"] == 0


def test_teacher_action_contract_rejects_report_mutation() -> None:
    report = PRIVILEGED_DOOR_TEACHER_ACTION_CONTRACT.to_report()
    report["max_yawrate_deg_s"] = 229.0

    with pytest.raises(ValueError, match="SHA-256"):
        DoorTeacherActionContract.from_report(report)


def test_teacher_action_contract_rejects_boolean_schema() -> None:
    report = PRIVILEGED_DOOR_TEACHER_ACTION_CONTRACT.to_report()
    report["schema_version"] = True
    payload = {key: value for key, value in report.items() if key != "sha256"}
    report["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    with pytest.raises(ValueError, match="schema"):
        DoorTeacherActionContract.from_report(report)


def test_only_current_privileged_teacher_contract_is_accepted() -> None:
    report = PRIVILEGED_DOOR_TEACHER_ACTION_CONTRACT.to_report()
    report["contract_id"] = "fixed-door-unreviewed-teacher"
    payload = {key: value for key, value in report.items() if key != "sha256"}
    report["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    with pytest.raises(ValueError, match="not current"):
        door_teacher_action_contract_from_report(report)


def test_privileged_teacher_contract_is_bound_to_edge_v3_action_envelope() -> None:
    scales = {name: scale for name, _unit, scale, _frame in ACTION_SPECS}
    contract = PRIVILEGED_DOOR_TEACHER_ACTION_CONTRACT

    assert (
        contract.contract_id
        == "fixed-door-privileged-teacher-edge-v3-action-envelope"
    )
    assert contract.max_forward_speed_m_s == scales["vx"]
    assert contract.max_yawrate_deg_s == scales["yaw_rate"]


def test_no_fixed_door_live_authority_contract_remains() -> None:
    import flightrl.puffer4_door_contract as contracts

    assert not hasattr(contracts, "FIXED_DOOR_LIVE_SAFETY_CONTRACT")
    assert not hasattr(contracts, "CORRECTED_DOOR_ACTION_CONTRACT")
