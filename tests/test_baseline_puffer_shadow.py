from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from flightrl.sixdof.puffer_policy import PufferPolicyMetadata, PufferSixDofPolicy


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("baseline_puffer_shadow", ROOT / "scripts" / "crazyflie_baseline_puffer_shadow.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_baseline_puffer_shadow_dry_run_is_monitor_only(tmp_path: Path) -> None:
    checkpoint = tmp_path / "puffer.bin"
    output = tmp_path / "baseline_shadow.csv"
    policy = PufferSixDofPolicy(PufferPolicyMetadata(observation_dim=28, hidden_size=8, action_dim=4, num_layers=2))
    torch.save(policy.state_dict(), checkpoint)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "crazyflie_baseline_puffer_shadow.py"),
            "--checkpoint",
            str(checkpoint),
            "--dry-run",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "puffer_controls_drone=False" in result.stdout
    rows = list(csv.DictReader(output.open()))
    assert rows[0]["phase"] == "dry_run"
    assert rows[0]["baseline_controls_drone"] == "False"
    assert rows[0]["puffer_controls_drone"] == "False"
    assert rows[0]["controls_drone"] == "False"
    assert rows[0]["raw_puffer_output"] == "True"


def test_baseline_puffer_shadow_rejects_zero_live_telemetry() -> None:
    zeros = {key: 0.0 for key in MODULE.REQUIRED_LIVE_FIELDS}
    assert MODULE.has_required_live_telemetry(zeros) is True
    assert MODULE.has_plausible_live_telemetry(zeros) is False


def test_baseline_puffer_shadow_requires_measured_lift_for_takeoff() -> None:
    assert MODULE.has_takeoff_evidence({"sys.isFlying": 1.0, "stateEstimate.z": 0.02}, 0.075) is False
    assert MODULE.has_takeoff_evidence({"sys.isFlying": 0.0, "stateEstimate.z": 0.30}, 0.075) is False
    assert MODULE.has_takeoff_evidence({"sys.isFlying": 1.0, "stateEstimate.z": 0.30}, 0.075) is True
    assert (
        MODULE.has_takeoff_evidence(
            {"sys.isFlying": 1.0, "stateEstimate.z": 0.30, "range.zrange": 30.0},
            0.075,
        )
        is False
    )


def test_baseline_puffer_shadow_builds_inflight_vision_capture_command() -> None:
    args = SimpleNamespace(
        vision_frames=48,
        vision_frame_dir="artifacts/frames",
        vision_output="artifacts/vision.npz",
        vision_transport="udp",
        vision_host="192.168.4.1",
        vision_port=5000,
        vision_bind_port=5001,
        vision_policy_checkpoint="artifacts/checkpoints/vision.pt",
    )

    command = MODULE.vision_capture_command(args)

    assert command[0] == sys.executable
    assert command[1].endswith("scripts/capture_aideck_vision.py")
    assert command[command.index("--frames") + 1] == "48"
    assert command[command.index("--transport") + 1] == "udp"
    assert command[command.index("--bind-port") + 1] == "5001"
    assert command[command.index("--frame-dir") + 1] == "artifacts/frames"
    assert command[command.index("--output") + 1] == "artifacts/vision.npz"
    assert command[command.index("--policy-checkpoint") + 1] == "artifacts/checkpoints/vision.pt"
    assert (
        MODULE.has_takeoff_evidence(
            {
                "sys.isFlying": 1.0,
                "stateEstimate.z": 0.30,
                "range.zrange": 300.0,
                "stabilizer.roll": 20.0,
            },
            0.075,
        )
        is False
    )
