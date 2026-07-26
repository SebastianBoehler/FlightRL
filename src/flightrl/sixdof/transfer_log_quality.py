from __future__ import annotations

from dataclasses import dataclass
from math import radians
from typing import Any

import numpy as np

from flightrl.hardware.sixdof_live_replay import live_env_from_telemetry, target_from_telemetry
from flightrl.sixdof import SixDofCrazyflieEnv, teacher_actions


ACTION_COLUMNS = ("action_thrust", "action_roll_rate", "action_pitch_rate", "action_yaw_rate")


@dataclass(frozen=True, slots=True)
class SourceTeacherQualityConfig:
    task: str = "obstacle_avoidance"
    target: tuple[float, float, float] = (0.0, 0.0, 0.5)
    target_yaw_deg: float = 0.0
    min_samples: int = 20
    sign_min_abs: float = 0.02
    min_sign_agreement: float = 0.65


def score_source_teacher_quality(rows: list[dict[str, float]], config: SourceTeacherQualityConfig) -> dict[str, Any]:
    scored = [row for row in rows if all(column in row for column in ACTION_COLUMNS)]
    if not scored:
        return {"samples": 0, "gate": {"passed": True, "failures": []}}
    env = SixDofCrazyflieEnv(num_envs=1, seed=0, task=config.task)
    target = np.asarray(config.target, dtype=np.float32)
    target_yaw = radians(config.target_yaw_deg)
    logged = []
    teachers = []
    for row in scored:
        live_env_from_telemetry(env, row, target=target_from_telemetry(row, target), target_yaw=target_yaw)
        logged.append([float(row[column]) for column in ACTION_COLUMNS])
        teachers.append(teacher_actions(env, task=config.task)[0].copy())
    logged_arr = np.asarray(logged, dtype=np.float32)
    teacher_arr = np.asarray(teachers, dtype=np.float32)
    errors = logged_arr - teacher_arr
    signs = {
        "thrust": sign_agreement(logged_arr[:, 0], teacher_arr[:, 0], config.sign_min_abs),
        "roll_rate": sign_agreement(logged_arr[:, 1], teacher_arr[:, 1], config.sign_min_abs),
        "pitch_rate": sign_agreement(logged_arr[:, 2], teacher_arr[:, 2], config.sign_min_abs),
    }
    return {
        "samples": int(len(scored)),
        "l2_p95": float(np.quantile(np.linalg.norm(errors, axis=1), 0.95)),
        "logged_action_abs_max": float(np.max(np.abs(logged_arr))),
        "teacher_action_abs_max": float(np.max(np.abs(teacher_arr))),
        "sign_agreement": signs,
        "gate": source_teacher_quality_gate(len(scored), signs, config),
    }


def source_teacher_quality_gate(samples: int, signs: dict[str, float], config: SourceTeacherQualityConfig) -> dict[str, Any]:
    failures = []
    if samples < config.min_samples:
        failures.append("source_teacher_samples")
    for axis, score in signs.items():
        if score < config.min_sign_agreement:
            failures.append(f"source_teacher_{axis}_sign")
    return {"passed": not failures, "failures": failures}


def sign_agreement(actual: np.ndarray, expected: np.ndarray, min_abs: float) -> float:
    mask = np.abs(expected) > min_abs
    if not np.any(mask):
        return 1.0
    return float(np.mean(np.sign(actual[mask]) == np.sign(expected[mask])))
