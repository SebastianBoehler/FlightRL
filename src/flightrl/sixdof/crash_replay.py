from __future__ import annotations

from dataclasses import asdict, dataclass
from math import radians
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from flightrl.hardware.sixdof_live_replay import live_env_from_telemetry, target_from_telemetry, value
from flightrl.hardware.sixdof_raw_action import RawPufferActionConfig, raw_action_to_manual_setpoint
from flightrl.hardware.sixdof_target import latched_target
from flightrl.sixdof import SixDofCrazyflieEnv, teacher_actions
from flightrl.sixdof.action_targets import shape_action_targets
from flightrl.sixdof.puffer_observation import scale_previous_action_observation


CRASH_GROUPS = ("precontact_drift", "close_recovery", "unsafe_tail")
ACTION_COLUMNS = ("action_thrust", "action_roll_rate", "action_pitch_rate", "action_yaw_rate")


@dataclass(frozen=True, slots=True)
class CrashReplayConfig:
    task: str = "obstacle_avoidance"
    target: tuple[float, float, float] = (0.0, 0.0, 0.50)
    target_mode: str = "current_pose"
    target_shaping: str = "none"
    target_shaping_strength: float = 1.0
    target_yaw_deg: float = 0.0
    precontact_min_horizontal_m: float = 0.45
    min_precontact_horizontal_speed_m_s: float = 0.45
    close_range_m: float = 0.35
    min_zrange_m: float = 0.18
    min_state_height_m: float = 0.20
    max_state_height_m: float = 1.20
    max_speed_m_s: float = 3.0
    max_abs_tilt_deg: float = 35.0
    min_samples: int = 16
    min_group_samples: int = 8
    max_l2_p95: float = 0.55
    max_precontact_l2_p95: float = 0.50
    min_sign_agreement: float = 0.45
    max_action_abs: float = 0.95
    precontact_target_clip_abs: float = 0.65
    close_target_clip_abs: float = 0.85
    unsafe_target_clip_abs: float = 0.45
    precontact_weight: float = 2.0
    close_weight: float = 1.0
    unsafe_weight: float = 0.6
    sign_agreement_min_abs: float = 0.02


def build_crash_replay_dataset(rows: list[dict[str, float]], config: CrashReplayConfig) -> dict[str, Any]:
    records = list(replay_records(rows, config, policy=None))
    observations = np.asarray([item["observation"] for item in records], dtype=np.float32)
    targets = np.asarray([item["target_action"] for item in records], dtype=np.float32)
    groups = np.asarray([item["primary_group"] for item in records], dtype=object)
    weights = np.asarray([item["sample_weight"] for item in records], dtype=np.float32)
    setpoints = np.asarray([teacher_setpoint_columns(item["target_action"]) for item in records], dtype=np.float32)
    return {
        "observations": observations.reshape((0, 28)) if observations.size == 0 else observations,
        "target_actions": targets.reshape((0, 4)) if targets.size == 0 else targets,
        "teacher_setpoints": setpoints.reshape((0, 5)) if setpoints.size == 0 else setpoints,
        "primary_groups": groups,
        "sample_weights": weights,
        "summary": dataset_summary(records),
        "config": asdict(config),
    }


def score_crash_replay_policy(
    policy,
    rows: list[dict[str, float]],
    config: CrashReplayConfig,
    *,
    previous_action_observation_scale: float = 1.0,
) -> dict[str, Any]:
    records = list(replay_records(rows, config, policy=policy, previous_action_observation_scale=previous_action_observation_scale))
    groups = {"all": records} | {name: [item for item in records if name in item["groups"]] for name in CRASH_GROUPS}
    metrics = {name: group_metrics(items, config.sign_agreement_min_abs) for name, items in groups.items()}
    return {
        "samples": len(records),
        "groups": metrics,
        "gate": crash_gate(metrics, config),
        "config": asdict(config),
    }


def replay_records(
    rows: list[dict[str, float]],
    config: CrashReplayConfig,
    *,
    policy=None,
    previous_action_observation_scale: float = 1.0,
) -> Iterable[dict[str, Any]]:
    env = SixDofCrazyflieEnv(num_envs=1, seed=0, task=config.task)
    target = latched_target(rows, config.target, config.target_mode)
    target_yaw = radians(config.target_yaw_deg)
    previous_action = np.zeros(4, dtype=np.float32)
    with torch.no_grad():
        for row in rows:
            live_env_from_telemetry(env, row, target=target_from_telemetry(row, target), target_yaw=target_yaw)
            env.previous_action[0] = previous_action
            observation = env.observation().astype(np.float32)
            teacher_action = shape_action_targets(
                env,
                teacher_actions(env, task=config.task),
                config.target_shaping,
                config.target_shaping_strength,
            )[0].astype(np.float32)
            policy_observation = scale_previous_action_observation(observation, previous_action_observation_scale)
            action = policy(torch.from_numpy(policy_observation)).cpu().numpy()[0].astype(np.float32) if policy else None
            groups = classify_row(row, config)
            if groups:
                target_action = shaped_target_action(teacher_action, groups, config)
                yield {
                    "observation": observation[0].copy(),
                    "target_action": target_action.copy(),
                    "action": None if action is None else action.copy(),
                    "groups": groups,
                    "primary_group": groups[0],
                    "sample_weight": sample_weight(groups[0], config),
                    "row": row,
                }
            previous_action = (action if action is not None else logged_action(row, teacher_action)).copy()


def shaped_target_action(action: np.ndarray, groups: tuple[str, ...], config: CrashReplayConfig) -> np.ndarray:
    target = np.asarray(action, dtype=np.float32).copy()
    if "unsafe_tail" in groups:
        return np.clip(target, -config.unsafe_target_clip_abs, config.unsafe_target_clip_abs).astype(np.float32)
    if "close_recovery" in groups:
        return np.clip(target, -config.close_target_clip_abs, config.close_target_clip_abs).astype(np.float32)
    if "precontact_drift" in groups:
        return np.clip(target, -config.precontact_target_clip_abs, config.precontact_target_clip_abs).astype(np.float32)
    return target


def classify_row(row: dict[str, float], config: CrashReplayConfig) -> tuple[str, ...]:
    groups: list[str] = []
    if precontact_row(row, config) and horizontal_speed_m_s(row) > config.min_precontact_horizontal_speed_m_s:
        groups.append("precontact_drift")
    if min_horizontal_range_m(row) < config.close_range_m:
        groups.append("close_recovery")
    if unsafe_row(row, config):
        groups.append("unsafe_tail")
    return tuple(groups)


def sample_weight(primary_group: str, config: CrashReplayConfig) -> float:
    if primary_group == "precontact_drift":
        return config.precontact_weight
    if primary_group == "unsafe_tail":
        return config.unsafe_weight
    return config.close_weight


def crash_gate(metrics: dict[str, dict[str, Any]], config: CrashReplayConfig) -> dict[str, Any]:
    failures: list[str] = []
    all_group = metrics.get("all", {})
    precontact = metrics.get("precontact_drift", {})
    if all_group.get("samples", 0) < config.min_samples:
        failures.append("crash_samples")
    if all_group.get("l2_p95", 0.0) > config.max_l2_p95:
        failures.append("crash_l2_p95")
    if all_group.get("action_abs_max", 0.0) > config.max_action_abs:
        failures.append("crash_action_abs")
    for axis, score in all_group.get("sign_agreement", {}).items():
        if score < config.min_sign_agreement:
            failures.append(f"crash_{axis}_sign")
    if precontact.get("samples", 0) >= config.min_group_samples:
        if precontact.get("l2_p95", 0.0) > config.max_precontact_l2_p95:
            failures.append("crash_precontact_l2_p95")
    return {"passed": not failures, "failures": failures}


def group_metrics(records: list[dict[str, Any]], sign_min_abs: float = 0.02) -> dict[str, Any]:
    if not records:
        return {"samples": 0}
    actions = np.asarray([item["action"] for item in records], dtype=np.float32)
    targets = np.asarray([item["target_action"] for item in records], dtype=np.float32)
    errors = actions - targets
    l2 = np.linalg.norm(errors, axis=1)
    action_abs = np.abs(actions)
    return {
        "samples": int(len(records)),
        "l2_mean": float(np.mean(l2)),
        "l2_p95": float(np.quantile(l2, 0.95)),
        "action_abs_p95": float(np.quantile(action_abs, 0.95)),
        "action_abs_max": float(np.max(action_abs)),
        "saturation_fraction": float(np.mean(action_abs >= 0.95)),
        "target_abs_max": float(np.max(np.abs(targets))),
        "sign_agreement": {
            "thrust": sign_agreement(actions[:, 0], targets[:, 0], sign_min_abs),
            "roll_rate": sign_agreement(actions[:, 1], targets[:, 1], sign_min_abs),
            "pitch_rate": sign_agreement(actions[:, 2], targets[:, 2], sign_min_abs),
        },
    }


def dataset_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {name: sum(name in item["groups"] for item in records) for name in CRASH_GROUPS}
    counts["all"] = len(records)
    return {"samples": len(records), "group_counts": counts}


def write_crash_replay_dataset(path: str | Path, dataset: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        observations=dataset["observations"],
        target_actions=dataset["target_actions"],
        teacher_setpoints=dataset["teacher_setpoints"],
        primary_groups=dataset["primary_groups"],
        sample_weights=dataset["sample_weights"],
    )


def teacher_setpoint_columns(action: np.ndarray) -> np.ndarray:
    setpoint = raw_action_to_manual_setpoint(action, RawPufferActionConfig())
    return np.asarray(
        [
            setpoint.thrust_percent,
            setpoint.roll_rate_deg_s,
            setpoint.pitch_rate_deg_s,
            setpoint.commander_pitch_rate_deg_s,
            setpoint.yaw_rate_deg_s,
        ],
        dtype=np.float32,
    )


def logged_action(row: dict[str, float], fallback: np.ndarray) -> np.ndarray:
    if all(column in row for column in ACTION_COLUMNS):
        return np.asarray([value(row, column) for column in ACTION_COLUMNS], dtype=np.float32)
    return np.asarray(fallback, dtype=np.float32)


def precontact_row(row: dict[str, float], config: CrashReplayConfig) -> bool:
    z = value(row, "stateEstimate.z", fallback=None)
    return (
        config.min_state_height_m <= z <= config.max_state_height_m
        and range_m(row, "range.zrange") >= config.min_zrange_m
        and min_horizontal_range_m(row) >= config.precontact_min_horizontal_m
    )


def unsafe_row(row: dict[str, float], config: CrashReplayConfig) -> bool:
    return (
        value(row, "sys.isTumbled") > 0.0
        or value(row, "sys.canfly", fallback=None) <= 0.0
        or speed_m_s(row) > config.max_speed_m_s
        or tilt_abs_deg(row) > config.max_abs_tilt_deg
        or range_m(row, "range.zrange") < config.min_zrange_m
    )


def min_horizontal_range_m(row: dict[str, float]) -> float:
    return min(range_m(row, key) for key in ("range.front", "range.back", "range.left", "range.right"))


def range_m(row: dict[str, float], key: str) -> float:
    raw = value(row, key, fallback=None)
    if raw <= 0.0 or not np.isfinite(raw):
        return 4.0
    return 4.0 if raw >= 32000.0 else raw / 1000.0


def horizontal_speed_m_s(row: dict[str, float]) -> float:
    return float(np.linalg.norm([value(row, "stateEstimate.vx"), value(row, "stateEstimate.vy")]))


def speed_m_s(row: dict[str, float]) -> float:
    return float(np.linalg.norm([value(row, "stateEstimate.vx"), value(row, "stateEstimate.vy"), value(row, "stateEstimate.vz")]))


def tilt_abs_deg(row: dict[str, float]) -> float:
    values = (
        value(row, "stabilizer.roll", fallback="stateEstimate.roll"),
        value(row, "stabilizer.pitch", fallback="stateEstimate.pitch"),
        value(row, "stateEstimate.roll", fallback=None),
        value(row, "stateEstimate.pitch", fallback=None),
    )
    return max(abs(item) for item in values if np.isfinite(item))


def sign_agreement(actual: np.ndarray, expected: np.ndarray, min_abs: float = 0.02) -> float:
    mask = np.abs(expected) > min_abs
    if not np.any(mask):
        return 1.0
    return float(np.mean(np.sign(actual[mask]) == np.sign(expected[mask])))
