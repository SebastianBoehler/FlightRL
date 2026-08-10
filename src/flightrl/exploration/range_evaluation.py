from __future__ import annotations

from math import atan2, pi
from typing import Callable

import numpy as np
import torch

from .range_curriculum import collect_range_natural_counterfactual_batch
from .range_challenge_evaluation import evaluate_range_obstacle_challenge
from .range_env import RangeExplorationEnv
from .range_policy import RangeExplorationActorCritic


RANGE_EVALUATION_SCHEMA = "flightrl.range_exploration.evaluation.v5"


def classical_frontier_action(observation: np.ndarray) -> np.ndarray:
    value = np.asarray(observation, dtype=np.float32)
    if value.shape != (4106,):
        raise ValueError("classical frontier observation must have 4106 values")
    frontier = value[:4096].reshape(4, 32, 32)[3]
    cells = np.argwhere(frontier > 0.5)
    if len(cells) == 0:
        return np.asarray((0.0, 0.35), dtype=np.float32)
    offsets = cells - np.asarray((16, 16))
    index = int(np.argmin(np.sum(offsets * offsets, axis=1)))
    row, column = cells[index]
    forward_m = (16.0 - float(row)) * 0.20
    left_m = (16.0 - float(column)) * 0.20
    bearing = atan2(left_m, max(0.01, forward_m))
    yaw = float(np.clip(bearing / (pi / 2.0), -1.0, 1.0))
    front_m = float(value[4096]) * 4.0
    front_valid = bool(value[4100])
    forward = 0.70 if abs(bearing) < 0.35 and front_valid and front_m > 0.40 else 0.0
    return np.asarray((forward, yaw), dtype=np.float32)


def evaluate_range_candidate(
    model: RangeExplorationActorCritic,
    *,
    seeds: tuple[int, ...],
    horizon: int,
) -> dict[str, object]:
    if not seeds or type(horizon) is not int or horizon <= 0:
        raise ValueError("range evaluation requires seeds and a positive horizon")
    modes = {
        mode: _policy_rollouts(model, seeds, horizon, mode)
        for mode in (
            "clean",
            "range_masked",
            "map_masked",
            "stress",
        )
    }
    baselines = {
        "stationary_scan": _controller_rollouts(
            seeds, horizon, lambda _obs: np.asarray((0.0, 0.35), dtype=np.float32)
        ),
        "classical_frontier": _controller_rollouts(
            seeds, horizon, classical_frontier_action
        ),
    }
    counterfactuals = range_counterfactual_checks(model)
    obstacle_challenge = evaluate_range_obstacle_challenge(model, horizon=horizon)
    checks = derive_range_evaluation_checks(
        modes, baselines, counterfactuals, obstacle_challenge
    )
    return {
        "schema": RANGE_EVALUATION_SCHEMA,
        "scope": "held_out_closed_loop_range_exploration",
        "seeds": list(seeds),
        "horizon": horizon,
        "modes": modes,
        "baselines": baselines,
        "counterfactuals": counterfactuals,
        "obstacle_challenge": obstacle_challenge,
        "checks": checks,
        "simulation_gate_passed": all(checks.values()),
        "actor_observation_contains_truth": False,
        "actor_receives_selected_frontier": False,
        "authority": {
            "training": False,
            "shadow": False,
            "deployment": False,
            "flight": False,
        },
    }


def derive_range_evaluation_checks(
    modes: dict[str, object],
    baselines: dict[str, object],
    counterfactuals: dict[str, bool],
    obstacle_challenge: dict[str, object],
) -> dict[str, bool]:
    clean = modes["clean"]
    return {
        "beats_stationary_and_classical_coverage": all(
            clean["mean_final_objective"] > baseline["mean_final_objective"]
            and clean["mean_objective_auc"] > baseline["mean_objective_auc"]
            for baseline in baselines.values()
        ),
        "zero_clean_collisions": clean["collision_rate"] == 0.0,
        "zero_clean_safety_terminals": clean["safety_terminal_rate"] == 0.0,
        "dedicated_obstacle_challenge": (
            obstacle_challenge["challenge_rate"] == 1.0
            and obstacle_challenge["escape_rate"] == 1.0
            and obstacle_challenge["collision_rate"] == 0.0
            and obstacle_challenge["safety_terminal_rate"] == 0.0
        ),
        "range_causal": clean["mean_final_coverage"]
        > modes["range_masked"]["mean_final_coverage"] + 0.01,
        "map_causal": clean["mean_final_coverage"]
        > modes["map_masked"]["mean_final_coverage"] + 0.01,
        "stress_collision_free": modes["stress"]["collision_rate"] == 0.0
        and modes["stress"]["safety_terminal_rate"] == 0.0,
        "front_obstacle_response": counterfactuals["front_obstacle_response"],
    }


def _policy_rollouts(
    model: RangeExplorationActorCritic,
    seeds: tuple[int, ...],
    horizon: int,
    mode: str,
) -> dict[str, object]:
    episodes = []
    model.eval()
    for seed in seeds:
        env = RangeExplorationEnv(
            seed=seed,
            maximum_episode_steps=horizon,
            stress=mode == "stress",
        )
        observation, _ = env.reset(seed=seed)
        coverage = []
        visited = []
        objectives = []
        collision = False
        safety_terminal = False
        minimum_front_m = 4.0
        path_m = 0.0
        for _step in range(horizon):
            minimum_front_m = min(minimum_front_m, float(observation[4096]) * 4.0)
            actor_observation = _ablate(observation, mode)
            before = env.truth_pose
            with torch.no_grad():
                action, _value = model.forward_step(
                    torch.from_numpy(actor_observation[None, :])
                )
            observation, _reward, terminated, truncated, info = env.step(
                action[0].cpu().numpy()
            )
            after = env.truth_pose
            path_m += float(np.hypot(after.x_m - before.x_m, after.y_m - before.y_m))
            coverage.append(float(info["coverage_fraction"]))
            visited.append(float(info["visited_fraction"]))
            objectives.append(0.35 * visited[-1] + 0.65 * coverage[-1])
            collision = collision or bool(info["collision"])
            safety_terminal = safety_terminal or bool(info["safety_terminal"])
            if terminated or truncated:
                break
        episodes.append(
            {
                "seed": seed,
                "final_coverage": coverage[-1] if coverage else 0.0,
                "coverage_auc": float(np.mean(coverage)) if coverage else 0.0,
                "final_visited": visited[-1] if visited else 0.0,
                "final_objective": objectives[-1] if objectives else 0.0,
                "objective_auc": float(np.mean(objectives)) if objectives else 0.0,
                "path_length_m": path_m,
                "collision": collision,
                "safety_terminal": safety_terminal,
                "minimum_front_range_m": minimum_front_m,
                "front_challenge": minimum_front_m < 0.65,
            }
        )
    return _summarize_episodes(episodes)


def _controller_rollouts(
    seeds: tuple[int, ...],
    horizon: int,
    controller: Callable[[np.ndarray], np.ndarray],
) -> dict[str, object]:
    episodes = []
    for seed in seeds:
        env = RangeExplorationEnv(seed=seed, maximum_episode_steps=horizon, stress=False)
        observation, _ = env.reset(seed=seed)
        coverage = []
        visited = []
        objectives = []
        collision = False
        safety_terminal = False
        minimum_front_m = 4.0
        path_m = 0.0
        for _step in range(horizon):
            minimum_front_m = min(minimum_front_m, float(observation[4096]) * 4.0)
            before = env.truth_pose
            observation, _reward, terminated, truncated, info = env.step(
                controller(observation)
            )
            after = env.truth_pose
            path_m += float(np.hypot(after.x_m - before.x_m, after.y_m - before.y_m))
            coverage.append(float(info["coverage_fraction"]))
            visited.append(float(info["visited_fraction"]))
            objectives.append(0.35 * visited[-1] + 0.65 * coverage[-1])
            collision = collision or bool(info["collision"])
            safety_terminal = safety_terminal or bool(info["safety_terminal"])
            if terminated or truncated:
                break
        episodes.append(
            {
                "seed": seed,
                "final_coverage": coverage[-1] if coverage else 0.0,
                "coverage_auc": float(np.mean(coverage)) if coverage else 0.0,
                "final_visited": visited[-1] if visited else 0.0,
                "final_objective": objectives[-1] if objectives else 0.0,
                "objective_auc": float(np.mean(objectives)) if objectives else 0.0,
                "path_length_m": path_m,
                "collision": collision,
                "safety_terminal": safety_terminal,
                "minimum_front_range_m": minimum_front_m,
                "front_challenge": minimum_front_m < 0.65,
            }
        )
    return _summarize_episodes(episodes)


def _summarize_episodes(episodes: list[dict[str, object]]) -> dict[str, object]:
    return {
        "episodes": episodes,
        "mean_final_coverage": float(np.mean([row["final_coverage"] for row in episodes])),
        "mean_coverage_auc": float(np.mean([row["coverage_auc"] for row in episodes])),
        "mean_final_visited": float(np.mean([row["final_visited"] for row in episodes])),
        "mean_final_objective": float(np.mean([row["final_objective"] for row in episodes])),
        "mean_objective_auc": float(np.mean([row["objective_auc"] for row in episodes])),
        "mean_path_length_m": float(np.mean([row["path_length_m"] for row in episodes])),
        "collision_rate": float(np.mean([row["collision"] for row in episodes])),
        "safety_terminal_rate": float(
            np.mean([row["safety_terminal"] for row in episodes])
        ),
        "challenge_rate": float(np.mean([row["front_challenge"] for row in episodes])),
    }


def _ablate(
    observation: np.ndarray,
    mode: str,
) -> np.ndarray:
    value = observation.copy()
    if mode == "range_masked":
        value[4096:4104] = 0.0
    elif mode == "map_masked":
        value[:4096] = 0.0
    return value


def range_counterfactual_checks(
    model: RangeExplorationActorCritic,
) -> dict[str, bool]:
    observations, _targets = collect_range_natural_counterfactual_batch(
        seed=36_701,
        base_count=1,
    )
    with torch.no_grad():
        actions, _value = model.forward_step(torch.from_numpy(observations))
    values = actions.cpu().numpy()
    return {
        "mirrored_frontier_direction": bool(values[0, 1] > 0.05 and values[1, 1] < -0.05),
        "front_obstacle_response": bool(values[2, 0] + 0.05 < values[0, 0]),
    }
