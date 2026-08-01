from __future__ import annotations

import hashlib
import json

import pytest

from flightrl.puffer4_config import Puffer4ExportSettings
from flightrl.puffer4_door_evidence_age_contract import (
    FIXED_DOOR_EVIDENCE_AGE_CONTRACT,
    approved_door_evidence_age_contract_from_report,
)
from flightrl.puffer4_door_sections import build_fixed_door_teacher_sections


def test_evidence_age_contract_drives_privileged_teacher_environment() -> None:
    sections = build_fixed_door_teacher_sections(Puffer4ExportSettings())

    assert sections["env"]["control_dt"] == pytest.approx(1.0 / 65.0)
    assert sections["env"]["maximum_evidence_age_s"] == 1.0
    FIXED_DOOR_EVIDENCE_AGE_CONTRACT.verify_env(sections["env"])


def test_evidence_age_contract_rejects_rehashed_unapproved_runtime() -> None:
    report = FIXED_DOOR_EVIDENCE_AGE_CONTRACT.to_report()
    report["maximum_evidence_age_s"] = 1.1
    payload = {key: value for key, value in report.items() if key != "sha256"}
    report["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    with pytest.raises(ValueError, match="approved"):
        approved_door_evidence_age_contract_from_report(report)


def test_evidence_age_contract_fails_closed_on_environment_drift() -> None:
    env = build_fixed_door_teacher_sections(Puffer4ExportSettings())["env"]
    env["maximum_evidence_age_s"] = 1.1

    with pytest.raises(ValueError, match="maximum_evidence_age_s"):
        FIXED_DOOR_EVIDENCE_AGE_CONTRACT.verify_env(env)
