from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from flightrl.sixdof.signal_evidence import RANGE_SIGNALS, REPLAY_STATE_SIGNALS


def write_ready_inputs(tmp_path: Path) -> dict:
    checkpoint = str(write_text(tmp_path / "ready.pt", "checkpoint"))
    bundle = str(write_text(tmp_path / "ready.edge-v3.bin", "bundle"))
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
        {
            "summary": {"passed": True, "failures": [], "gain_imbalance": 0.02},
            "simulator_priors": {
                "present": True,
                "mean_slope_rpm_per_power": 0.45,
                "relative_motor_gains": {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0},
            },
        },
    )
    noise = write_json(tmp_path / "noise.json", {"summary": {"stationary_noise_ready": True, "failures": [], "rows": 100, "duration_s": 60.0, "sample_rate_hz": 10.0, "max_position_span_m": 0.01, "max_attitude_span_deg": 0.2}, "signals": noise_signals()})
    latency = write_json(tmp_path / "latency.json", {"summary": {"latency_ready": True, "failures": [], "accepted_pairs": 2, "median_latency_s": 0.04}})
    quality = write_json(tmp_path / "quality.json", {"summary": {"replay_calibration_ready": True, "failures": [], "rows": 500, "duration_s": 10.0, "sample_rate_hz": 50.0}})
    replay = write_json(
        tmp_path / "replay.json",
        {
            "aligned": {
                "samples": 500,
                "overlap_duration_s": 10.0,
                "signals": {
                    name: {
                        "samples": 500,
                        "rmse": 20.0 if name in RANGE_SIGNALS else 0.1,
                    }
                    for name in (*REPLAY_STATE_SIGNALS, *RANGE_SIGNALS)
                },
            }
        },
    )
    deployment = write_json(
        tmp_path / "deploy.json",
        readiness(checkpoint, ready=True, deployment=True, bundle=bundle),
    )
    sim = write_json(tmp_path / "sim.json", readiness(checkpoint, ready=True))
    room = write_json(tmp_path / "room.json", {"summary": {"mapping_ready": True, "failures": [], "point_count": 100, "duration_s": 10.0}, "room_estimate": {"width_m": 2.0, "depth_m": 3.0, "height_m": 2.5}})
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


def readiness(
    checkpoint: str,
    *,
    ready: bool,
    deployment: bool = False,
    bundle: str | None = None,
) -> dict:
    scope = (
        {
            "schema": "flightrl.edge_v3.deployment_readiness.v1",
            "target": "ai_deck_gap8",
            "evidence_scope": "edge_deployment",
            "deployment_authority": True,
        }
        if deployment
        else {"evidence_scope": "desktop_development", "deployment_authority": False}
    )
    record = {
        "task": "obstacle_avoidance",
        "tasks": ["obstacle_avoidance"],
        "controller": "policy",
        "label": "ready",
        "checkpoint": checkpoint,
        "ready": ready,
        "failures": [],
    }
    if deployment:
        assert bundle is not None
        record.update(
            {
                "checkpoint_identity": file_identity(Path(checkpoint)),
                "bundle": bundle,
                "bundle_identity": file_identity(Path(bundle)),
            }
        )
    return {
        **scope,
        "summary": {"total": 1, "ready": int(ready), "blocked": int(not ready)},
        "records": [record],
        "global_evidence": {
            "training_throughput": {"present": True, "valid": True, "best_total_sps": {"total_sps": 1000.0}},
            "puffer_export": {"present": True, "passed": True, "env_name": "flightrl_sixdof", "checks": [{"passed": True}]},
        },
    }


def noise_signals() -> dict:
    keys = [
        "stateEstimate.x", "stateEstimate.y", "stateEstimate.z",
        "stabilizer.roll", "stabilizer.pitch", "stabilizer.yaw",
        "acc.x", "acc.y", "acc.z", "gyro.x", "gyro.y", "gyro.z",
        "range.front", "range.back", "range.left", "range.right", "range.up", "range.zrange",
    ]
    return {key: {"samples": 100, "valid_ratio": 1.0, "std": 0.01} for key in keys}


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


def file_identity(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
