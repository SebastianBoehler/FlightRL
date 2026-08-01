from __future__ import annotations

import json
import subprocess
import sys

from flightrl.evidence_scope import EDGE_DEPLOYMENT_VERIFIER_MISSING
from flightrl.sim2real.evidence_gap import build_evidence_gap_report


def test_gap_report_blocks_current_missing_evidence(tmp_path) -> None:
    pipeline = write_json(
        tmp_path / "pipeline.json",
        {
            "transfer_approved": False,
            "hardware_approved_checkpoints": 0,
            "blocking_items": ["m3_motor_issue", "motor_bench_failed", "replay_comparison_failed"],
        },
    )

    report = build_evidence_gap_report(pipeline)

    assert report["enough_for_one_step_transfer"] is False
    assert report["decision"] == "blocked"
    assert report["categories"]["hardware_repair"] == ["m3_motor_issue"]
    assert report["categories"]["actuator_model"] == ["motor_bench_failed"]
    assert report["pipeline"]["sha256"]


def test_gap_report_requires_edge_v3_bundle_even_when_old_fields_claim_approval(tmp_path) -> None:
    pipeline = write_json(
        tmp_path / "pipeline.json",
        {"transfer_approved": True, "hardware_approved_checkpoints": 1, "blocking_items": []},
    )

    report = build_evidence_gap_report(pipeline)

    assert report["enough_for_one_step_transfer"] is False
    assert report["decision"] == "blocked"
    assert report["transfer_approved"] is False
    assert report["hardware_approved_checkpoints"] == 0
    assert report["claimed_transfer_approved"] is True
    assert report["categories"]["policy_deployment"] == [
        EDGE_DEPLOYMENT_VERIFIER_MISSING
    ]


def test_gap_report_cli_writes_json_and_markdown(tmp_path) -> None:
    pipeline = write_json(
        tmp_path / "pipeline.json",
        {"transfer_approved": False, "hardware_approved_checkpoints": 0, "blocking_items": ["latency_failed"]},
    )
    output = tmp_path / "gap.json"

    result = subprocess.run(
        [sys.executable, "scripts/build_sim2real_evidence_gap.py", "--pipeline", str(pipeline), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "decision=blocked" in result.stdout
    assert output.exists()
    assert output.with_suffix(".md").exists()


def write_json(path, data):
    path.write_text(json.dumps(data))
    return path
