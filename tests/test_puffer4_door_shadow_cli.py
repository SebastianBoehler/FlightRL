from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/crazyflie_door_puffer_shadow.py"
CHECKPOINT = (
    ROOT
    / "artifacts/puffer_fixed_door_d1_v59_fresh_control_bc1m"
    / "flightrl_fixed_door_d1_seed11_1048576.bin"
)
REPORT = CHECKPOINT.with_suffix(".reevaluation.json")


def test_exact_morning_shadow_cli_dry_run_writes_bound_nonactuating_rows(
    tmp_path: Path,
) -> None:
    output = tmp_path / "shadow.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--checkpoint",
            str(CHECKPOINT),
            "--training-report",
            str(REPORT),
            "--prompt",
            "interior door",
            "--threshold",
            "0.25",
            "--duration-s",
            "20",
            "--output",
            str(output),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    with output.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["controls_drone"] == "False"
    assert row["monitor_only"] == "True"
    assert row["yaw_only_projected_forward_m_s"] == "0.0"
    assert float(row["yaw_only_projected_yawrate_deg_s"]) <= 8.0
    assert row["executed_previous_forward_normalized"] == "0.0"
    assert row["executed_previous_yaw_normalized"] == "0.0"
    assert row["stream_dropped_frames"] == "0"
    identity = json.loads(row["shadow_run_identity_json"])
    assert identity["checkpoint"]["sha256"] == (
        "f676d12b9d37c27f4cc62f99beceec8f30e74c88be8564cb242c23755e202cce"
    )
    assert identity["inference_device"] == "mps"
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert summary["shadow_run_identity"] == identity
    assert summary["yaw_only_projection_contract_passed"] is True


def test_shadow_cli_rejects_changed_detector_before_capture(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--checkpoint",
            str(CHECKPOINT),
            "--training-report",
            str(REPORT),
            "--prompt",
            "door",
            "--output",
            str(tmp_path / "shadow.csv"),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "approved shadow detector runtime" in result.stderr
    assert not (tmp_path / "shadow.csv").exists()


@pytest.mark.parametrize("existing", ("csv", "summary"))
def test_shadow_cli_exclusively_rejects_existing_output_before_evidence_load(
    tmp_path: Path,
    existing: str,
) -> None:
    output = tmp_path / "shadow.csv"
    occupied = (
        output
        if existing == "csv"
        else output.with_suffix(".summary.json")
    )
    occupied.write_text("keep me")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--checkpoint",
            str(tmp_path / "missing.bin"),
            "--training-report",
            str(tmp_path / "missing.json"),
            "--output",
            str(output),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert occupied.read_text() == "keep me"
    if existing == "summary":
        assert not output.exists()
