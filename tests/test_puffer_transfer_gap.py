from __future__ import annotations

from flightrl.sixdof.transfer_gap import candidate_gap_summary


def test_candidate_gap_summary_prioritizes_command_transform_and_precontact_drift() -> None:
    candidate = {
        "passed": False,
        "sim": {"python": {"gate": {"passed": False, "failures": ["open_space_horizontal_speed_p95"]}}},
        "live_logs": {
            "raw": {
                "failed_source": True,
                "shadow": {"gate": {"passed": False, "failures": ["shadow_pitch_rate_sign"]}},
                "command_gate": {"passed": False, "failures": ["commander_pitch_sign_mismatch", "safe_roll_pitch_rate_p95"]},
                "source_failure_evidence": {
                    "passed": False,
                    "failures": ["source_precontact_drift"],
                    "source": {
                        "precontact_horizontal_speed_max_m_s": 1.4,
                        "horizontal_min_mm": 640.0,
                        "tilt_max_abs_deg": 57.0,
                    },
                },
                "crash_replay": {"gate": {"passed": False, "failures": ["crash_l2_p95"]}},
            }
        },
    }

    summary = candidate_gap_summary(candidate)

    assert summary["counts"] == {"sim": 1, "shadow": 1, "command": 2, "crash_replay": 1, "source": 1}
    assert summary["live_log_failures"]["raw"]["precontact_horizontal_speed_max_m_s"] == 1.4
    assert summary["primary_blockers"][:3] == [
        "sim:python:open_space_horizontal_speed_p95",
        "command_transform:raw:commander_pitch_sign_mismatch",
        "source_precontact_drift:raw:source_precontact_drift",
    ]


def test_candidate_gap_summary_passed_candidate_has_no_blockers() -> None:
    summary = candidate_gap_summary({"passed": True, "sim": {"python": {"gate": {"passed": True, "failures": []}}}, "live_logs": {}})

    assert summary["primary_blockers"] == ["none"]
    assert summary["counts"]["sim"] == 0
