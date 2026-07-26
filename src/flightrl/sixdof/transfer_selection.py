from __future__ import annotations

from math import radians
from typing import Any

import numpy as np
import torch

from flightrl.hardware.direct_raw_gate import DirectRawGateThresholds, evaluate_direct_raw_replay
from flightrl.hardware.sixdof_live_replay import live_env_from_telemetry, target_from_telemetry
from flightrl.hardware.sixdof_target import latched_target
from flightrl.sixdof import SixDofCrazyflieEnv, teacher_actions
from flightrl.sixdof.action_targets import shape_action_targets
from flightrl.sixdof.crash_replay import classify_row, logged_action
from flightrl.sixdof.puffer_observation import scale_previous_action_observation
from flightrl.sixdof.transfer_test import (
    LiveLogCase,
    TransferTestConfig,
    bottom_range,
    crash_config_from_transfer,
    live_shadow_report,
    load_live_rows,
    raw_shadow_rows,
    top_range,
)


PreparedTransferLog = tuple[LiveLogCase, list[dict[str, float]]]


def prepare_transfer_selection(specs: list[str], *, failed_source: bool = False) -> list[PreparedTransferLog]:
    prepared = []
    for spec in specs:
        label, path = split_label_path(spec)
        case = LiveLogCase(label=label, path=path, failed_source=failed_source)
        prepared.append((case, load_live_rows(path)))
    return prepared


def transfer_shadow_selection_metrics(policy, prepared: list[PreparedTransferLog], config: TransferTestConfig) -> dict[str, Any]:
    if not prepared:
        return {}
    failure_count = 0
    l2_excess = 0.0
    action_excess = 0.0
    sign_gap = 0.0
    command_failure_count = 0
    command_saturation_excess = 0.0
    command_rate_excess = 0.0
    labels = {}
    for case, rows in prepared:
        report = live_shadow_report(policy, rows, case, config)
        command = evaluate_direct_raw_replay(
            raw_shadow_rows(policy, rows, config),
            DirectRawGateThresholds(min_safe_rows=config.min_command_safe_rows, require_source_health=False),
        )
        gate = report["gate"]
        all_group = report["groups"].get("all", {})
        safe = command.get("safe", {})
        failure_count += len(gate["failures"])
        command_failure_count += len(command["failures"])
        l2_excess += max(0.0, all_group.get("l2_p95", 0.0) - config.max_shadow_l2_p95)
        action_excess += max(0.0, all_group.get("action_abs_max", 0.0) - config.max_shadow_action_abs)
        command_saturation_excess += max(0.0, safe.get("action_saturation_fraction", 0.0) - 0.08)
        command_rate_excess += max(0.0, max(safe.get("roll_rate_abs_p95", 0.0), safe.get("pitch_rate_abs_p95", 0.0)) - 220.0) / 220.0
        for score in all_group.get("sign_agreement", {}).values():
            sign_gap += max(0.0, config.min_shadow_sign_agreement - score)
        labels[case.label] = {
            "failures": gate["failures"],
            "command_failures": command["failures"],
            "l2_p95": all_group.get("l2_p95", 0.0),
            "action_abs_max": all_group.get("action_abs_max", 0.0),
            "command_action_saturation_fraction": safe.get("action_saturation_fraction", 0.0),
            "command_roll_rate_abs_p95": safe.get("roll_rate_abs_p95", 0.0),
            "command_pitch_rate_abs_p95": safe.get("pitch_rate_abs_p95", 0.0),
            "sign_agreement": all_group.get("sign_agreement", {}),
        }
    return {
        "transfer_shadow_failure_count": float(failure_count),
        "transfer_shadow_l2_excess": float(l2_excess),
        "transfer_shadow_action_excess": float(action_excess),
        "transfer_shadow_sign_gap": float(sign_gap),
        "transfer_command_failure_count": float(command_failure_count),
        "transfer_command_saturation_excess": float(command_saturation_excess),
        "transfer_command_rate_excess": float(command_rate_excess),
        "transfer_shadow_labels": labels,
    }


def transfer_shadow_selection_score(metrics: dict[str, Any]) -> float:
    if not metrics:
        return 0.0
    return -(
        3.0 * float(metrics["transfer_shadow_failure_count"])
        + float(metrics["transfer_shadow_l2_excess"])
        + 2.0 * float(metrics["transfer_shadow_action_excess"])
        + 2.0 * float(metrics["transfer_shadow_sign_gap"])
        + 1.5 * float(metrics.get("transfer_command_failure_count", 0.0))
        + 4.0 * float(metrics.get("transfer_command_saturation_excess", 0.0))
        + 2.0 * float(metrics.get("transfer_command_rate_excess", 0.0))
    )


def build_transfer_replay(
    prepared: list[PreparedTransferLog],
    config: TransferTestConfig,
    target_shaping: str = "none",
    target_shaping_strength: float = 1.0,
) -> dict[str, torch.Tensor] | None:
    if not prepared:
        return None
    env = SixDofCrazyflieEnv(num_envs=1, seed=0, task=config.task, sensor_profile=config.sensor_profile)
    target_yaw = radians(config.target_yaw_deg)
    observations = []
    targets = []
    vertical = []
    weights = []
    sequence_start = []
    crash_config = crash_config_from_transfer(config)
    excluded_source_rows = 0
    source_rows = 0
    for case, rows in prepared:
        target = latched_target(rows, config.target, config.target_mode)
        case_observations = []
        case_targets = []
        case_vertical = []
        case_sequence_start = []
        previous_action = np.zeros(4, dtype=np.float32)
        first_kept_row = True
        for row in rows:
            source_rows += 1
            live_env_from_telemetry(env, row, target=target_from_telemetry(row, target), target_yaw=target_yaw)
            env.previous_action[0] = previous_action
            teacher_action = shape_action_targets(env, teacher_actions(env, task=config.task), target_shaping, target_shaping_strength)[0].copy()
            if case.failed_source and classify_row(row, crash_config):
                excluded_source_rows += 1
                previous_action = logged_action(row, teacher_action)
                continue
            observation = scale_previous_action_observation(env.observation(), config.previous_action_observation_scale)
            case_observations.append(observation[0].copy())
            case_targets.append(teacher_action)
            case_vertical.append(is_vertical_replay_row(case, row))
            case_sequence_start.append(first_kept_row)
            first_kept_row = False
            previous_action = logged_action(row, teacher_action)
        if not case_observations:
            continue
        observations.extend(case_observations)
        targets.extend(case_targets)
        vertical.extend(case_vertical)
        sequence_start.extend(case_sequence_start)
        weights.extend([1.0 / len(case_observations)] * len(case_observations))
    if not observations:
        return None
    return {
        "observations": torch.tensor(np.asarray(observations), dtype=torch.float32),
        "target_actions": torch.tensor(np.asarray(targets), dtype=torch.float32),
        "vertical_mask": torch.tensor(vertical, dtype=torch.bool),
        "sequence_start": torch.tensor(sequence_start, dtype=torch.bool),
        "sample_weights": normalized_case_weights(weights),
        "source_rows": source_rows,
        "excluded_source_rows": excluded_source_rows,
    }


def numeric_metrics(metrics: dict) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def normalized_case_weights(weights: list[float]) -> torch.Tensor:
    raw = np.asarray(weights, dtype=np.float32)
    return torch.tensor(raw / max(float(raw.mean()), 1e-6), dtype=torch.float32)


def is_vertical_replay_row(case: LiveLogCase, row: dict[str, float]) -> bool:
    return "vertical" in case.label.lower() or min(top_range(row), bottom_range(row)) < 0.45


def split_label_path(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError("transfer selection logs must use LABEL:PATH")
    label, path = spec.split(":", 1)
    if not label or not path:
        raise ValueError("transfer selection logs must use LABEL:PATH")
    return label, path
