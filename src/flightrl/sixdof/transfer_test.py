from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from math import radians
from pathlib import Path
from typing import Any

import numpy as np
import torch

from flightrl.hardware.direct_raw_gate import DirectRawGateThresholds, evaluate_direct_raw_replay
from flightrl.hardware.sixdof_live_replay import live_env_from_telemetry, target_from_telemetry, value
from flightrl.hardware.sixdof_raw_action import RawPufferActionConfig, raw_action_to_manual_setpoint
from flightrl.hardware.sixdof_target import latched_target
from flightrl.sixdof import SixDofCrazyflieEnv, teacher_actions
from flightrl.sixdof.crash_replay import CrashReplayConfig, classify_row, score_crash_replay_policy
from flightrl.sixdof.puffer_evaluation import PufferEvalConfig, evaluate_puffer_backends
from flightrl.sixdof.puffer_observation import scale_previous_action_observation
from flightrl.sixdof.puffer_policy import load_puffer_sixdof_policy


@dataclass(frozen=True, slots=True)
class LiveLogCase:
    label: str
    path: str
    failed_source: bool = False


@dataclass(frozen=True, slots=True)
class TransferTestConfig:
    task: str = "obstacle_avoidance"
    steps: int = 300
    num_envs: int = 128
    seed: int = 707
    reset_profile: str = "obstacle_hover_live"
    physics_profile: str | None = None
    sensor_profile: str | None = None
    domain_randomization: str | None = None
    disturbance_profile: str | None = None
    max_open_space_horizontal_speed_p95_m_s: float = 0.75
    target: tuple[float, float, float] = (0.0, 0.0, 0.50)
    target_mode: str = "current_pose"
    crash_target_shaping: str = "none"
    crash_target_shaping_strength: float = 1.0
    target_yaw_deg: float = 0.0
    min_shadow_samples: int = 100
    min_group_samples: int = 50
    max_shadow_l2_p95: float = 0.55
    max_shadow_close_l2_p95: float = 0.60
    min_shadow_sign_agreement: float = 0.45
    sign_agreement_min_abs: float = 0.02
    max_shadow_action_abs: float = 0.95
    min_command_safe_rows: int = 80
    previous_action_observation_scale: float = 0.25


def evaluate_transfer_candidate(checkpoint: str, live_logs: list[LiveLogCase], config: TransferTestConfig) -> dict[str, Any]:
    policy = load_puffer_sixdof_policy(checkpoint)
    sim_config = PufferEvalConfig(
        task=config.task,
        backend="both",
        steps=config.steps,
        num_envs=config.num_envs,
        seed=config.seed,
        reset_profile=config.reset_profile,
        physics_profile=config.physics_profile,
        sensor_profile=config.sensor_profile,
        domain_randomization=config.domain_randomization,
        disturbance_profile=config.disturbance_profile,
        max_open_space_horizontal_speed_p95_m_s=config.max_open_space_horizontal_speed_p95_m_s,
        previous_action_observation_scale=config.previous_action_observation_scale,
    )
    report = {
        "checkpoint": checkpoint,
        "config": asdict(config),
        "sim": evaluate_puffer_backends(policy, sim_config),
        "live_logs": {},
    }
    for case in live_logs:
        rows = load_live_rows(case.path)
        shadow = live_shadow_report(policy, rows, case, config)
        command_rows = raw_shadow_rows(policy, rows, config)
        command_gate = evaluate_direct_raw_replay(
            command_rows,
            DirectRawGateThresholds(
                min_safe_rows=config.min_command_safe_rows,
                require_source_health=False,
                require_commander_pitch_sign=True,
            ),
        )
        item = {
            "path": case.path,
            "failed_source": case.failed_source,
            "shadow": shadow,
            "command_gate": command_gate,
        }
        if case.failed_source:
            item["source_failure_evidence"] = evaluate_direct_raw_replay(
                rows,
                DirectRawGateThresholds(min_safe_rows=0, require_commander_pitch_sign=False),
            )
            item["crash_replay"] = score_crash_replay_policy(policy, rows, crash_config_from_transfer(config), previous_action_observation_scale=config.previous_action_observation_scale)
        report["live_logs"][case.label] = item
    report["passed"] = transfer_passed(report)
    return report


def transfer_passed(report: dict[str, Any]) -> bool:
    sim_passed = all(item.get("gate", {}).get("passed", False) for item in report["sim"].values())
    live_passed = all(
        item["shadow"]["gate"]["passed"] and item["command_gate"]["passed"]
        and item.get("crash_replay", {"gate": {"passed": True}})["gate"]["passed"]
        for item in report["live_logs"].values()
    )
    return bool(sim_passed and live_passed)


def live_shadow_report(policy, rows: list[dict[str, float]], case: LiveLogCase, config: TransferTestConfig) -> dict[str, Any]:
    pairs = shadow_pairs(policy, rows, config)
    scored, excluded = shadow_scored_pairs(pairs, case, config)
    groups = shadow_groups(scored)
    if case.failed_source:
        groups["source_all"] = pairs
        groups["source_excluded"] = excluded
    metrics = {name: group_metrics(group, config.sign_agreement_min_abs) for name, group in groups.items()}
    return {
        "label": case.label,
        "samples": len(pairs),
        "scored_samples": len(scored),
        "excluded_source_samples": len(excluded),
        "groups": metrics,
        "gate": shadow_gate(metrics, config),
    }


def shadow_scored_pairs(
    pairs: list[tuple[np.ndarray, np.ndarray, dict[str, float]]],
    case: LiveLogCase,
    config: TransferTestConfig,
) -> tuple[list[tuple[np.ndarray, np.ndarray, dict[str, float]]], list[tuple[np.ndarray, np.ndarray, dict[str, float]]]]:
    if not case.failed_source:
        return pairs, []
    crash_config = crash_config_from_transfer(config)
    scored = []
    excluded = []
    for pair in pairs:
        if classify_row(pair[2], crash_config):
            excluded.append(pair)
        else:
            scored.append(pair)
    return scored, excluded


def shadow_groups(pairs: list[tuple[np.ndarray, np.ndarray, dict[str, float]]]) -> dict[str, list[tuple[np.ndarray, np.ndarray, dict[str, float]]]]:
    return {
        "all": pairs,
        "close_lt_32cm": [pair for pair in pairs if min_range(pair[2]) < 0.32],
        "top_lt_45cm": [pair for pair in pairs if top_range(pair[2]) < 0.45],
        "vertical_lt_35cm": [pair for pair in pairs if min(top_range(pair[2]), bottom_range(pair[2])) < 0.35],
    }


def crash_config_from_transfer(config: TransferTestConfig) -> CrashReplayConfig:
    return CrashReplayConfig(task=config.task, target=config.target, target_mode=config.target_mode, target_shaping=config.crash_target_shaping, target_shaping_strength=config.crash_target_shaping_strength, target_yaw_deg=config.target_yaw_deg)


def shadow_pairs(policy, rows: list[dict[str, float]], config: TransferTestConfig) -> list[tuple[np.ndarray, np.ndarray, dict[str, float]]]:
    env = SixDofCrazyflieEnv(num_envs=1, seed=0, task=config.task, sensor_profile=config.sensor_profile)
    target = latched_target(rows, config.target, config.target_mode)
    target_yaw = radians(config.target_yaw_deg)
    previous_action = np.zeros(4, dtype=np.float32)
    pairs = []
    with torch.no_grad():
        for row in rows:
            live_env_from_telemetry(env, row, target=target_from_telemetry(row, target), target_yaw=target_yaw)
            env.previous_action[0] = previous_action
            observation = scale_previous_action_observation(env.observation().astype(np.float32), config.previous_action_observation_scale)
            action = policy(torch.from_numpy(observation)).cpu().numpy()[0]
            teacher = teacher_actions(env, task=config.task)[0]
            pairs.append((action, teacher, row))
            previous_action[:] = action
    return pairs


def raw_shadow_rows(policy, rows: list[dict[str, float]], config: TransferTestConfig) -> list[dict[str, float]]:
    env = SixDofCrazyflieEnv(num_envs=1, seed=0, task=config.task, sensor_profile=config.sensor_profile)
    target = latched_target(rows, config.target, config.target_mode)
    target_yaw = radians(config.target_yaw_deg)
    previous_action = np.zeros(4, dtype=np.float32)
    raw_config = RawPufferActionConfig()
    output = []
    with torch.no_grad():
        for row in rows:
            live_env_from_telemetry(env, row, target=target_from_telemetry(row, target), target_yaw=target_yaw)
            env.previous_action[0] = previous_action
            observation = scale_previous_action_observation(env.observation().astype(np.float32), config.previous_action_observation_scale)
            action = policy(torch.from_numpy(observation)).cpu().numpy()[0]
            setpoint = raw_action_to_manual_setpoint(action, raw_config)
            previous_action[:] = action
            output.append(
                {
                    **row,
                    "raw_puffer_output": True,
                    "raw_control_active": False,
                    "action_thrust": float(action[0]),
                    "action_roll_rate": float(action[1]),
                    "action_pitch_rate": float(action[2]),
                    "action_yaw_rate": float(action[3]),
                    "roll_rate_deg_s": setpoint.roll_rate_deg_s,
                    "pitch_rate_deg_s": setpoint.pitch_rate_deg_s,
                    "commander_pitch_rate_deg_s": setpoint.commander_pitch_rate_deg_s,
                    "yaw_rate_deg_s": setpoint.yaw_rate_deg_s,
                    "thrust_percent": setpoint.thrust_percent,
                }
            )
    return output


def shadow_gate(groups: dict[str, dict[str, Any]], config: TransferTestConfig) -> dict[str, Any]:
    failures = []
    all_group = groups.get("all", {})
    if all_group.get("samples", 0) < config.min_shadow_samples:
        failures.append("shadow_samples")
    if all_group.get("l2_p95", 0.0) > config.max_shadow_l2_p95:
        failures.append("shadow_l2_p95")
    if all_group.get("action_abs_max", 0.0) > config.max_shadow_action_abs:
        failures.append("shadow_action_abs")
    for axis, score in all_group.get("sign_agreement", {}).items():
        if score < config.min_shadow_sign_agreement:
            failures.append(f"shadow_{axis}_sign")
    close = groups.get("close_lt_32cm", {})
    if close.get("samples", 0) >= config.min_group_samples and close.get("l2_p95", 0.0) > config.max_shadow_close_l2_p95:
        failures.append("shadow_close_l2_p95")
    return {"passed": not failures, "failures": failures}


def group_metrics(group: list[tuple[np.ndarray, np.ndarray, dict[str, float]]], sign_min_abs: float = 0.02) -> dict[str, Any]:
    if not group:
        return {"samples": 0}
    actions = np.asarray([item[0] for item in group], dtype=np.float32)
    teachers = np.asarray([item[1] for item in group], dtype=np.float32)
    errors = actions - teachers
    l2 = np.linalg.norm(errors, axis=1)
    return {
        "samples": len(group),
        "l2_p95": float(np.quantile(l2, 0.95)),
        "action_abs_max": float(np.max(np.abs(actions))),
        "saturation_fraction": float(np.mean(np.abs(actions) > 0.95)),
        "sign_agreement": {
            "thrust": sign_agreement(actions[:, 0], teachers[:, 0], sign_min_abs),
            "roll_rate": sign_agreement(actions[:, 1], teachers[:, 1], sign_min_abs),
            "pitch_rate": sign_agreement(actions[:, 2], teachers[:, 2], sign_min_abs),
        },
    }


def load_live_rows(path: str | Path) -> list[dict[str, float]]:
    parsed = []
    latest: dict[str, float] = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            latest.update({key: parse_float(raw) for key, raw in row.items() if raw != ""})
            parsed.append(dict(latest))
    return parsed


def parse_float(raw: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def min_range(row: dict[str, float]) -> float:
    if "min_horizontal_range_m" in row:
        return value(row, "min_horizontal_range_m")
    return min(live_range_m(row, key) for key in ("range.front", "range.back", "range.left", "range.right"))


def top_range(row: dict[str, float]) -> float:
    return live_range_m(row, "range.up")


def bottom_range(row: dict[str, float]) -> float:
    return live_range_m(row, "range.zrange")


def live_range_m(row: dict[str, float], key: str) -> float:
    raw = value(row, key)
    if raw <= 0.0 or not np.isfinite(raw):
        return 4.0
    return 4.0 if raw >= 32000.0 else raw / 1000.0


def sign_agreement(actual: np.ndarray, expected: np.ndarray, min_abs: float = 0.02) -> float:
    mask = np.abs(expected) > min_abs
    if not np.any(mask):
        return 1.0
    return float(np.mean(np.sign(actual[mask]) == np.sign(expected[mask])))
