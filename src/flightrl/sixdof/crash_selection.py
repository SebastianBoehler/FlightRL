from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .puffer_observation import scale_previous_action_observation

def crash_replay_selection_metrics(
    policy,
    replay: dict[str, torch.Tensor] | None,
    *,
    action_abs_limit: float,
    previous_action_observation_scale: float = 1.0,
) -> dict[str, float]:
    if replay is None or len(replay["observations"]) == 0:
        return {}
    with torch.no_grad():
        observations = scale_previous_action_observation(replay["observations"], previous_action_observation_scale)
        prediction = policy(observations)
    targets = replay["target_actions"]
    l2 = torch.linalg.norm(prediction - targets, dim=1)
    action_abs = torch.abs(prediction)
    metrics = {
        "crash_replay_l2_p95": float(torch.quantile(l2, 0.95)),
        "crash_replay_action_abs_max": float(torch.max(action_abs)),
        "crash_replay_saturation_fraction": float(torch.mean((action_abs > action_abs_limit).float())),
    }
    primary_groups = replay.get("primary_groups")
    if primary_groups is not None:
        precontact = torch.tensor(np.asarray(primary_groups) == "precontact_drift", dtype=torch.bool)
        if bool(torch.any(precontact)):
            metrics["crash_replay_precontact_l2_p95"] = float(torch.quantile(l2[precontact], 0.95))
    return metrics


def crash_replay_selection_score(metrics: dict[str, Any], *, action_abs_limit: float) -> float:
    if not metrics:
        return 0.0
    l2_excess = max(0.0, float(metrics["crash_replay_l2_p95"]) - 0.55)
    precontact_l2 = float(metrics.get("crash_replay_precontact_l2_p95", metrics["crash_replay_l2_p95"]))
    precontact_excess = max(0.0, precontact_l2 - 0.50)
    action_excess = max(0.0, float(metrics["crash_replay_action_abs_max"]) - action_abs_limit)
    saturation = float(metrics["crash_replay_saturation_fraction"])
    return -float(np.clip(l2_excess + precontact_excess + 2.0 * action_excess + saturation, 0.0, 10.0))


def load_replay_npz(path: str | None) -> dict[str, torch.Tensor] | None:
    if not path:
        return None
    data = np.load(path, allow_pickle=True)
    replay = {
        "observations": torch.tensor(data["observations"], dtype=torch.float32),
        "target_actions": torch.tensor(data["target_actions"], dtype=torch.float32),
    }
    if "sample_weights" in data:
        replay["sample_weights"] = torch.tensor(data["sample_weights"], dtype=torch.float32)
    if "primary_groups" in data:
        replay["primary_groups"] = data["primary_groups"]
    return replay


def replay_samples(path: str | None) -> int:
    replay = load_replay_npz(path)
    return 0 if replay is None else int(len(replay["observations"]))
