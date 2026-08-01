from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import torch

from flightrl.sixdof import SixDofPolicy, build_checkpoint_payload

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_sixdof_candidate_matrix",
    ROOT / "scripts" / "build_sixdof_candidate_matrix.py",
)
assert SPEC and SPEC.loader
MATRIX = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MATRIX
SPEC.loader.exec_module(MATRIX)


def suite_record(label: str, checkpoint: Path) -> dict:
    return {
        "label": label,
        "controller": "policy",
        "checkpoint": str(checkpoint),
        "tasks": ["position_yaw"],
        "gate": {"passed": True, "failures": []},
        "per_task_gate": {"position_yaw": {"passed": True, "failures": []}},
        "metrics": {
            "mean_completed_fraction": 1.0,
            "mean_survival_fraction": 1.0,
            "mean_position_error_m": 0.1,
            "mean_yaw_error_rad": 0.05,
            "yaw_error_p95_rad": 0.07,
            "min_clearance_m": 0.2,
            "clearance_p01_m": 0.2,
        },
    }


def write_checkpoint(
    path: Path,
    *,
    tasks: tuple[str, ...] = ("position_yaw",),
    controller: str = "policy",
    residual_scale: float = 0.0,
) -> None:
    input_dim = 28 + (len(tasks) if len(tasks) > 1 else 0)
    torch.save(
        build_checkpoint_payload(
            state_dict=SixDofPolicy(hidden_size=16, input_dim=input_dim).state_dict(),
            tasks=tasks,
            hidden_size=16,
            controller=controller,
            residual_scale=residual_scale,
        ),
        path,
    )


def identity(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def record(
    label: str,
    tasks: list[str],
    *,
    completed: float,
    position_error: float,
    yaw_error: float | None = None,
) -> dict:
    return {
        "label": label,
        "checkpoint": f"{label}.pt",
        "tasks": tasks,
        "controller": "policy",
        "passed": True,
        "failures": [],
        "mean_completed_fraction": completed,
        "mean_survival_fraction": completed,
        "mean_position_error_m": position_error,
        "mean_yaw_error_rad": yaw_error,
        "yaw_error_p95_rad": yaw_error,
        "clearance_p01_m": 0.2,
        "desktop_parity": {"present": True, "passed": True},
        "desktop_latency": {"present": True, "per_sample_us": 4.0},
    }
