from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from flightrl.puffer4_door_policy_contract import (
    door_policy_architecture_from_report,
    door_policy_contract_report,
    verify_door_policy_contract,
)


def _rehashed(report: dict) -> dict:
    payload = {key: value for key, value in report.items() if key != "sha256"}
    report["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return report


def _existing_v1_report() -> dict:
    report = deepcopy(door_policy_contract_report(hidden_size=96, num_layers=1))
    report["contract_id"] = "fixed-door-recurrent-policy-v1"
    report["schema_version"] = 1
    report["observation"]["frame"][
        "self_mask"
    ] = "lower_center_fill_post_quantization_global_mean"
    return _rehashed(report)


def test_policy_contract_v2_freezes_observation_and_recurrence_layout() -> None:
    report = door_policy_contract_report(hidden_size=96, num_layers=1)

    verify_door_policy_contract(report, hidden_size=96, num_layers=1)

    assert report["contract_id"] == "fixed-door-recurrent-policy-v2"
    assert report["schema_version"] == 2
    assert report["observation"]["total_floats"] == 9248
    assert report["observation"]["deployable_floats"] == 9242
    assert report["observation"]["segments"]["previous_action"] == [9231, 9233]
    assert (
        report["observation"]["frame"]["self_mask"]
        == "upper_corner_wedges_fill_post_quantization_global_mean"
    )
    assert report["recurrence"]["kind"] == "MinGRU"
    assert report["recurrence"]["hidden_size"] == 96
    assert report["recurrence"]["terminal_reset"] == "zero_after_terminal_step"


def test_policy_contract_rejects_mutation_or_wrong_recurrence() -> None:
    report = door_policy_contract_report(hidden_size=96, num_layers=1)
    report["observation"]["total_floats"] = 9247

    with pytest.raises(ValueError, match="SHA-256"):
        verify_door_policy_contract(report, hidden_size=96, num_layers=1)

    report = door_policy_contract_report(hidden_size=96, num_layers=1)
    with pytest.raises(ValueError, match="does not match"):
        verify_door_policy_contract(report, hidden_size=64, num_layers=1)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("contract_id", "fixed-door-recurrent-policy-v3"),
        ("schema_version", 3),
        ("schema_version", [2]),
        ("self_mask", "upper_corner_rectangles"),
    ),
)
def test_policy_contract_rejects_self_consistent_unapproved_payload(
    field: str,
    value: object,
) -> None:
    report = deepcopy(door_policy_contract_report(hidden_size=96, num_layers=1))
    if field == "self_mask":
        report["observation"]["frame"][field] = value
    else:
        report[field] = value
    _rehashed(report)

    with pytest.raises(ValueError, match="approved"):
        door_policy_architecture_from_report(report)


def test_policy_architecture_decodes_from_existing_v1_payload() -> None:
    report = _existing_v1_report()
    original = report.copy()

    architecture = door_policy_architecture_from_report(report)

    assert architecture.hidden_size == 96
    assert architecture.num_layers == 1
    assert report["sha256"] == (
        "ad6fa58f50a1c0754d572643a9d7affe65f3e73d4d814c51030c733588ef8058"
    )
    assert report == original
