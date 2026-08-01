from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

from flightrl.sixdof.puffer_readiness import REQUIRED_CHECKS
from flightrl.sixdof.signal_evidence import (
    NATIVE_STATE_SIGNALS,
    RANGE_SIGNALS,
    REPLAY_STATE_SIGNALS,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_sixdof_readiness_report",
    ROOT / "scripts" / "build_sixdof_readiness_report.py",
)
assert SPEC and SPEC.loader
READINESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = READINESS
SPEC.loader.exec_module(READINESS)


def candidate_record(
    *,
    label: str = "candidate",
    checkpoint: str = "candidate.pt",
    model: Path | None = None,
    passed: bool = True,
    parity: bool = True,
    latency: float | None = 9.0,
    tasks: list[str] | None = None,
) -> dict:
    checkpoint_identity = (
        file_identity(Path(checkpoint)) if Path(checkpoint).exists() else None
    )
    model_identity = file_identity(model) if model is not None else None
    return {
        "label": label,
        "controller": "policy",
        "checkpoint": checkpoint,
        "tasks": tasks or ["obstacle_avoidance"],
        "passed": passed,
        "failures": [] if passed else ["position_error"],
        "mean_completed_fraction": 1.0,
        "mean_position_error_m": 0.1,
        "mean_yaw_error_rad": 0.05,
        "yaw_error_p95_rad": 0.07,
        "clearance_p01_m": 0.5,
        "desktop_parity": {
            "present": parity,
            "passed": parity,
            "max_abs_error": 0.0 if parity else None,
            "checkpoint": checkpoint_identity,
            "model": model_identity,
            "evidence_scope": "desktop_cpu_only",
            "deployment_authority": False,
        },
        "desktop_latency": {
            "present": latency is not None,
            "checkpoint": checkpoint_identity,
            "model": None,
            "evidence_scope": "desktop_cpu_only",
            "deployment_authority": False,
            **(
                {
                    "per_sample_us": latency,
                    "samples_per_second": 1_000_000.0 / latency,
                }
                if latency is not None
                else {}
            ),
        },
        "per_task_gate": {
            task: {"passed": True, "failures": []}
            for task in (tasks or ["obstacle_avoidance"])
        },
        "checkpoint_meta": {"controller": "policy", "hidden_size": 128},
    }


def room_report(*, mapping_ready: bool) -> dict:
    return {
        "summary": {
            "mapping_ready": mapping_ready,
            "failures": [],
            "point_count": 100,
            "duration_s": 10.0,
        },
        "room_estimate": {
            "width_m": 2.0,
            "depth_m": 3.0,
            "height_m": 2.5,
            "warnings": [],
        },
    }


def native_report(*, state_rmse: float, range_rmse: float, mismatches: int = 0) -> dict:
    aggregate_signals = {
        name: {
            "samples": 10,
            "rmse": range_rmse if name in RANGE_SIGNALS else state_rmse,
            "mae": range_rmse if name in RANGE_SIGNALS else state_rmse,
            "max_abs": range_rmse if name in RANGE_SIGNALS else state_rmse,
            "worst_profile": "broad",
        }
        for name in (*NATIVE_STATE_SIGNALS, *RANGE_SIGNALS)
    }
    profile_signals = {
        name: {key: value for key, value in metrics.items() if key != "worst_profile"}
        for name, metrics in aggregate_signals.items()
    }
    return {
        "aligned": {
            "samples": 10,
            "overlap_duration_s": 1.0,
            "signals": aggregate_signals,
        },
        "reset_profiles": ["broad"],
        "profiles": [
            {
                "reset_profile": "broad",
                "samples": 10,
                "duration_s": 1.0,
                "terminal_mismatches": mismatches,
                "truncation_mismatches": 0,
                "signals": profile_signals,
            }
        ],
    }


def replay_report(*, state_rmse: float, range_rmse: float, overlap: float = 2.0) -> dict:
    return {
        "aligned": {
            "samples": 20,
            "overlap_duration_s": overlap,
            "signals": {
                name: {
                    "samples": 20,
                    "rmse": range_rmse if name in RANGE_SIGNALS else state_rmse,
                }
                for name in (*REPLAY_STATE_SIGNALS, *RANGE_SIGNALS)
            },
        }
    }


def residual_sweep_report() -> dict:
    return {
        "run": True,
        "thresholds": {"max_teacher_action_l2_mean": 0.02},
        "summary": {
            "total": 1,
            "completed": 1,
            "best": {
                "name": "scale005",
                "checkpoint": "residual.pt",
                "passed": True,
                "mean_completed_fraction": 1.0,
                "mean_position_error_m": 0.19,
                "mean_yaw_error_rad": 0.2,
                "teacher_action_l2_mean": 0.001,
            },
        },
    }


def throughput_report(
    total_sps: float = 123456.0,
    *,
    controller: str = "policy",
    tasks: list[str] | None = None,
) -> dict:
    return {
        "controller": controller,
        "residual_scale": 0.05 if controller == "teacher_residual" else 0.0,
        "tasks": tasks or ["obstacle_avoidance"],
        "records": [{"name": "base"}],
        "summary": {
            "total": 1,
            "best_total_sps": {
                "name": "base",
                "total_sps": total_sps,
                "num_envs": 256,
                "horizon": 32,
                "hidden_size": 128,
            },
        },
    }


def puffer_export_report(*, passed: bool = True) -> dict:
    return {
        "passed": passed,
        "env_name": "flightrl_sixdof",
        "checks": [
            {"name": name, "passed": passed, "failures": []}
            for name in sorted(REQUIRED_CHECKS)
        ],
        "config": {
            "base": {"env_name": "flightrl_sixdof"},
            "env": {"task_id": "1"},
            "policy": {"hidden_size": "128"},
        },
        "files": {
            "binding.c": {"exists": True, "bytes": 100, "lines": 10},
        },
    }


def profile_matrix(*, passed: bool) -> dict:
    return {
        "profiles": ["position_yaw_recovery", "broad"],
        "records": [
            {
                "label": "candidate",
                "controller": "policy",
                "checkpoint": "candidate.pt",
                "tasks": ["position_yaw"],
                "passed_all_profiles": passed,
                "missing_profiles": [],
                "failures_by_profile": {} if passed else {"broad": ["completion"]},
                "profiles": {
                    "position_yaw_recovery": {
                        "passed": True,
                        "failures": [],
                        "mean_survival_fraction": 0.7,
                        "mean_completed_fraction": 0.3,
                        "mean_position_error_m": 2.0,
                        "mean_yaw_error_rad": 0.2,
                        "clearance_p01_m": 0.05,
                    },
                    "broad": {
                        "passed": passed,
                        "failures": [] if passed else ["completion"],
                        "mean_survival_fraction": 0.7,
                        "mean_completed_fraction": 0.3,
                        "mean_position_error_m": 2.0,
                        "mean_yaw_error_rad": 0.2,
                        "clearance_p01_m": 0.05,
                    },
                },
                "worst_survival_fraction": 0.7,
                "worst_completed_fraction": 0.3,
                "worst_position_error_m": 2.0,
                "worst_clearance_p01_m": 0.05,
                "worst_yaw_error_rad": 0.2,
            }
        ],
    }


def argparse_like(**kwargs):
    return type("Args", (), kwargs)()


def file_identity(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
