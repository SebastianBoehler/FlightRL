from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from flightrl.sim2real.pipeline import build_pipeline, output_paths


ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_builds_complete_artifact_chain(tmp_path: Path) -> None:
    paths = write_ready_inputs(tmp_path)
    outputs = output_paths(tmp_path / "out", "ready")

    report = build_pipeline(outputs=outputs, **paths)

    assert report["transfer_approved"] is True
    assert report["hardware_approved_checkpoints"] == 1
    assert report["inputs"]["hardware_config"]["exists"] is True
    assert len(report["inputs"]["hardware_config"]["sha256"]) == 64
    assert report["inputs"]["hardware_config"]["size_bytes"] > 0
    assert report["inputs"]["live_scripts"][0]["exists"] is True
    assert len(report["inputs"]["live_scripts"][0]["sha256"]) == 64
    assert outputs["pipeline"].exists()
    assert outputs["checkpoint_manifest"].exists()
    assert "evidence_gap" in report["artifacts"]


def test_pipeline_cli_rebuilds_blocked_current_style_chain(tmp_path: Path) -> None:
    paths = write_ready_inputs(tmp_path)
    output_dir = tmp_path / "out"
    blockers = write_json(tmp_path / "blockers.json", {"blockers": []})

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_sim2real_pipeline.py",
            "--label",
            "cli",
            "--output-dir",
            str(output_dir),
            "--hardware-config",
            str(paths["hardware_config"]),
            "--base-config",
            str(paths["base_config"]),
            "--output-config",
            str(paths["output_config"]),
            "--motor-calibration",
            str(paths["motor_calibration"]),
            "--stationary-noise",
            str(paths["stationary_noise"]),
            "--hardware-latency",
            str(paths["hardware_latency"]),
            "--calibration-quality",
            str(paths["calibration_quality"]),
            "--deployment-readiness",
            str(paths["deployment_readiness"]),
            "--replay-comparison",
            str(paths["replay_comparison"]),
            "--motor-bench",
            str(paths["motor_bench"]),
            "--sim-readiness",
            str(paths["sim_readiness"]),
            "--room-report",
            str(paths["room_report"]),
            "--live-script",
            str(paths["live_scripts"][0]),
            "--hardware-blockers-file",
            str(blockers),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "transfer_approved=True" in result.stdout
    assert (output_dir / "sim2real_pipeline_cli.json").exists()
    assert (output_dir / "sim2real_evidence_gap_cli.json").exists()


def test_pipeline_cli_uses_hardware_blocker_file_by_default(tmp_path: Path) -> None:
    paths = write_ready_inputs(tmp_path)
    blockers = write_json(tmp_path / "blockers.json", {"blockers": ["m3_motor_issue"]})
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_sim2real_pipeline.py",
            "--label",
            "blocked",
            "--output-dir",
            str(output_dir),
            "--hardware-config",
            str(paths["hardware_config"]),
            "--base-config",
            str(paths["base_config"]),
            "--output-config",
            str(paths["output_config"]),
            "--motor-calibration",
            str(paths["motor_calibration"]),
            "--stationary-noise",
            str(paths["stationary_noise"]),
            "--hardware-latency",
            str(paths["hardware_latency"]),
            "--calibration-quality",
            str(paths["calibration_quality"]),
            "--deployment-readiness",
            str(paths["deployment_readiness"]),
            "--replay-comparison",
            str(paths["replay_comparison"]),
            "--motor-bench",
            str(paths["motor_bench"]),
            "--sim-readiness",
            str(paths["sim_readiness"]),
            "--room-report",
            str(paths["room_report"]),
            "--live-script",
            str(paths["live_scripts"][0]),
            "--hardware-blockers-file",
            str(blockers),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    pipeline = json.loads((output_dir / "sim2real_pipeline_blocked.json").read_text())
    assert "transfer_approved=False" in result.stdout
    assert "m3_motor_issue" in pipeline["blocking_items"]
    assert pipeline["inputs"]["hardware_blockers_file"]["path"] == str(blockers)
    assert len(pipeline["inputs"]["hardware_blockers_file"]["sha256"]) == 64
    assert pipeline["inputs"]["hardware_blockers"] == ["m3_motor_issue"]


def write_ready_inputs(tmp_path: Path) -> dict:
    checkpoint = "artifacts/checkpoints/ready.pt"
    hardware = write_text(
        tmp_path / "measured.toml",
        """
[environment]
dt = 0.02
action_mode = "motor_quad"

[sim2real]
measured = true
source = "test"

[drone]
mass = 1.15
inertia = 0.09
arm_length = 0.23
drag = 0.14
angular_drag = 0.09
hover_thrust = 10.6
thrust_gain = 4.2
max_total_thrust = 19.5
max_pitch_torque = 2.2
actuator_tau = 0.11

[sensors]
include_noisy_state = true
""",
    )
    base = write_text(tmp_path / "base.toml", "[environment]\ndt = 0.02\n\n[drone]\nmass = 1.0\n")
    motor_cal = write_json(
        tmp_path / "motor_cal.json",
        {"summary": {"passed": True, "failures": [], "gain_imbalance": 0.02}, "simulator_priors": {"present": True, "relative_motor_gains": {"1": 1.0}}},
    )
    noise = write_json(tmp_path / "noise.json", {"summary": {"stationary_noise_ready": True, "failures": []}, "signals": noise_signals()})
    latency = write_json(tmp_path / "latency.json", {"summary": {"latency_ready": True, "failures": [], "median_latency_s": 0.04}})
    quality = write_json(tmp_path / "quality.json", {"summary": {"replay_calibration_ready": True, "failures": []}})
    replay = write_json(tmp_path / "replay.json", {"aligned": {"signals": {"stateEstimate.x": {"rmse": 0.1}, "range.front": {"rmse": 20.0}}}})
    deployment = write_json(tmp_path / "deploy.json", readiness(checkpoint, ready=True))
    sim = write_json(tmp_path / "sim.json", readiness(checkpoint, ready=True))
    room = write_json(tmp_path / "room.json", {"summary": {"mapping_ready": True, "failures": [], "point_count": 100}, "room_estimate": {"width_m": 2.0}})
    motor_bench = write_motor_bench(tmp_path / "motor.csv")
    live_script = write_text(tmp_path / "safe_live.py", 'parser.add_argument("--checkpoint")\nrequire_policy_approval(args.checkpoint, args.approval_manifest)\nmodules = require_cflib()\ncommander.send_stop_setpoint()\n')
    return {
        "hardware_config": hardware,
        "base_config": base,
        "output_config": tmp_path / "measured_sim.toml",
        "deployment_readiness": deployment,
        "sim_readiness": sim,
        "live_scripts": [live_script],
        "motor_calibration": motor_cal,
        "stationary_noise": noise,
        "hardware_latency": latency,
        "calibration_quality": quality,
        "replay_comparison": replay,
        "motor_bench": motor_bench,
        "room_report": room,
        "hardware_blockers": [],
    }


def readiness(checkpoint: str, *, ready: bool) -> dict:
    return {
        "summary": {"total": 1, "ready": int(ready), "blocked": int(not ready)},
        "records": [{"task": "obstacle_avoidance", "label": "ready", "checkpoint": checkpoint, "ready": ready, "failures": []}],
        "global_evidence": {"training_throughput": {"present": True}, "puffer_export": {"present": True, "passed": True}},
    }


def noise_signals() -> dict:
    return {key: {"std": 0.01} for key in ["stateEstimate.x", "stateEstimate.y", "stateEstimate.z", "range.front"]}


def write_motor_bench(path: Path) -> Path:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["motor", "power", "rpm", "motor_output", "motor_requested", "vbat"])
        writer.writeheader()
        for motor in range(1, 5):
            for power in [14000, 20000, 26000]:
                writer.writerow({"motor": motor, "power": power, "rpm": power + motor, "motor_output": power, "motor_requested": power, "vbat": 4.0})
    return path


def write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data))
    return path


def write_text(path: Path, text: str) -> Path:
    path.write_text(text.strip() + "\n")
    return path
