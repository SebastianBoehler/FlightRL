from __future__ import annotations

import pytest

from sixdof_readiness_test_support import (
    READINESS,
    argparse_like,
    candidate_record,
    native_report,
    profile_matrix,
    replay_report,
    room_report,
    throughput_report,
)


def test_readiness_report_rejects_bad_replay_comparison() -> None:
    args = argparse_like(
        require_replay_comparison=False,
        max_replay_state_rmse=0.5,
        max_replay_range_rmse_mm=300.0,
        min_replay_overlap_s=1.0,
    )
    replay = READINESS.compact_replay_comparison(
        replay_report(state_rmse=0.1, range_rmse=600.0), args
    )

    assert replay["passed"] is False
    assert "range_rmse" in replay["failures"]


def test_readiness_report_rejects_native_termination_mismatch() -> None:
    compact = READINESS.compact_native_parity(
        native_report(state_rmse=1e-8, range_rmse=0.1, mismatches=1), 1e-5, 1.0
    )

    assert compact["passed"] is False
    assert "termination_mismatch" in compact["failures"]


def test_readiness_report_can_require_training_throughput() -> None:
    evidence = {
        "room": {"present": True, "mapping_ready": True},
        "native_parity": {"present": True, "passed": True},
        "replay_comparison": {"present": False, "required": False, "passed": True},
        "training_throughput": READINESS.compact_training_throughput(
            throughput_report(total_sps=123.0)
        ),
    }

    record = READINESS.evaluate_record(
        ("obstacle_avoidance", candidate_record()), evidence, 50.0, True, 1000.0
    )

    assert record["ready"] is False
    assert "training_throughput_slow" in record["failures"]


def test_readiness_report_reports_missing_required_training_throughput() -> None:
    evidence = {
        "room": {"present": True, "mapping_ready": True},
        "native_parity": {"present": True, "passed": True},
        "replay_comparison": {"present": False, "required": False, "passed": True},
        "training_throughput": {"present": False},
    }

    record = READINESS.evaluate_record(
        ("obstacle_avoidance", candidate_record()), evidence, 50.0, True, 1.0
    )

    assert "training_throughput_missing" in record["failures"]


def test_readiness_candidates_include_multitask_after_single_tasks() -> None:
    matrix = {
        "best_by_task": {"position_yaw": candidate_record(label="single")},
        "best_multitask": candidate_record(
            label="multi", tasks=["position_yaw", "circle"]
        ),
    }

    assert [key for key, _ in READINESS.readiness_candidates(matrix)] == [
        "position_yaw",
        "multitask",
    ]


def test_readiness_rejects_nonfinite_global_evidence() -> None:
    native = native_report(state_rmse=float("nan"), range_rmse=0.1)
    replay = replay_report(state_rmse=0.1, range_rmse=float("inf"))
    args = argparse_like(
        require_replay_comparison=False,
        max_replay_state_rmse=0.5,
        max_replay_range_rmse_mm=300.0,
        min_replay_overlap_s=1.0,
    )

    assert READINESS.compact_native_parity(native, 1.0e-5, 1.0)["passed"] is False
    assert READINESS.compact_replay_comparison(replay, args)["passed"] is False
    throughput = READINESS.compact_training_throughput(
        throughput_report(total_sps=float("nan"))
    )
    assert READINESS.training_throughput_failures(
        throughput, require=True, min_total_sps=1.0
    ) == ["training_throughput_missing"]


def test_readiness_rejects_truthy_room_flag_and_invalid_threshold() -> None:
    room = room_report(mapping_ready=True)
    room["summary"]["mapping_ready"] = "true"

    assert READINESS.compact_room(room)["mapping_ready"] is False
    with pytest.raises(ValueError, match="max_desktop_latency_us"):
        READINESS.evaluate_record(
            ("obstacle_avoidance", candidate_record()),
            {
                "room": {"mapping_ready": True},
                "native_parity": {"passed": True},
                "replay_comparison": {"present": False, "required": False},
            },
            float("nan"),
        )


def test_readiness_profile_evidence_does_not_fall_back_by_label() -> None:
    profile = profile_matrix(passed=True)
    profile["records"][0]["checkpoint"] = "different.pt"
    compact = READINESS.compact_profile_matrix(profile)

    assert READINESS.profile_record(candidate_record(), compact) == {"present": False}


def test_readiness_rejects_unscoped_or_legacy_matrix() -> None:
    with pytest.raises(ValueError, match="scope"):
        READINESS.validate_matrix({"best_by_task": {}, "best_multitask": None})


def test_readiness_rejects_truthy_puffer_export_pass() -> None:
    compact = READINESS.compact_puffer_export(
        {
            "passed": "true",
            "env_name": "flightrl_sixdof",
            "checks": [{"passed": True}],
        }
    )

    assert compact["passed"] is False

    contradictory = READINESS.compact_puffer_export(
        {
            "passed": True,
            "env_name": "flightrl_sixdof",
            "checks": [{"passed": True, "failures": ["stale"]}],
        }
    )
    assert contradictory["passed"] is False


def test_readiness_rejects_incomplete_native_and_replay_signal_sets() -> None:
    native = native_report(state_rmse=0.0, range_rmse=0.0)
    native["aligned"]["signals"].pop("stateEstimate.y")
    args = argparse_like(
        require_replay_comparison=True,
        max_replay_state_rmse=0.5,
        max_replay_range_rmse_mm=300.0,
        min_replay_overlap_s=1.0,
    )
    replay = replay_report(state_rmse=0.0, range_rmse=0.0)
    replay["aligned"]["signals"].pop("range.back")

    assert READINESS.compact_native_parity(native, 1e-5, 1.0)["passed"] is False
    assert READINESS.compact_replay_comparison(replay, args)["passed"] is False


def test_readiness_rejects_declared_profiles_without_profile_evidence() -> None:
    report = profile_matrix(passed=True)
    del report["records"][0]["profiles"]["broad"]

    with pytest.raises(ValueError, match="profile evidence is incomplete"):
        READINESS.compact_profile_matrix(report)


def test_readiness_rejects_duplicate_profile_checkpoint_records() -> None:
    report = profile_matrix(passed=True)
    report["records"].append(dict(report["records"][0]))

    with pytest.raises(ValueError, match="duplicate checkpoint"):
        READINESS.compact_profile_matrix(report)


def test_readiness_rejects_unbound_training_throughput() -> None:
    evidence = {
        "room": {"present": True, "mapping_ready": True},
        "native_parity": {"present": True, "passed": True},
        "replay_comparison": {"present": False, "required": False, "passed": True},
        "training_throughput": READINESS.compact_training_throughput(
            throughput_report(controller="teacher_residual", tasks=["circle"])
        ),
    }

    record = READINESS.evaluate_record(
        ("obstacle_avoidance", candidate_record()),
        evidence,
        50.0,
        True,
        1.0,
    )

    assert "training_throughput_contract" in record["failures"]
