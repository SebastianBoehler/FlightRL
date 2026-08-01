from __future__ import annotations

from collections.abc import Mapping
from math import isclose, isfinite, sqrt

from flightrl.puffer4_edge_evaluation_counts import (
    exact_episode_count,
    native_fraction_count,
    subgroup_counts,
)


EDGE_EVALUATION_SCHEMA = "flightrl.edge_v3.closed_loop_evaluation.v2"
EDGE_STUDENT_GATE_THRESHOLDS = {
    "success_rate": (">=", 0.90),
    "collision_rate": ("<=", 0.02),
    "collision_rate_upper_95": ("<=", 0.05),
    "outside_fov_success_rate": (">=", 0.85),
    "outside_fov_episodes": (">=", 32.0),
    "episodes": (">=", 128.0),
    "action_rmse": ("<=", 0.25),
    "door_action_rmse": ("<=", 0.25),
    "reset_action_rmse": ("<=", 0.30),
    "reset_door_action_rmse": ("<=", 0.35),
    "reset_samples": (">=", 128.0),
    "lateral_action_abs_mean": ("<=", 0.10),
    "vertical_action_abs_mean": ("<=", 0.10),
    "lateral_action_abs_max": ("<=", 0.25),
    "vertical_action_abs_max": ("<=", 0.25),
    "grounding_visibility_precision": (">=", 0.80),
    "grounding_visibility_recall": (">=", 0.80),
    "grounding_visible_box_mae": ("<=", 0.15),
    "grounding_visible_samples": (">=", 1024.0),
    "grounding_absent_samples": (">=", 1024.0),
    "hidden_min": (">=", 0.0),
    "hidden_max": ("<=", 6.0),
}
EDGE_EVALUATION_PROFILES = (
    (
        "clean",
        31_001,
        61_001,
        {
            "obstacle_probability": 0.0,
            "camera_randomization": 0.0,
            "layout_diversity": 1.0,
        },
    ),
    (
        "mixed",
        32_001,
        62_001,
        {
            "obstacle_probability": 0.5,
            "camera_randomization": 1.0,
            "layout_diversity": 1.0,
        },
    ),
    (
        "obstacle",
        33_001,
        63_001,
        {
            "obstacle_probability": 1.0,
            "camera_randomization": 1.0,
            "layout_diversity": 1.0,
        },
    ),
)
_UNIT_INTERVAL_METRICS = frozenset(
    {
        "success_rate", "collision_rate", "collision_rate_upper_95",
        "outside_fov_success_rate",
        "lateral_action_abs_mean", "vertical_action_abs_mean",
        "lateral_action_abs_max", "vertical_action_abs_max",
        "grounding_visibility_precision", "grounding_visibility_recall",
    }
)
_SUBGROUP_REQUIREMENTS = {
    "layout_family": (4, 24.0, 0.85),
    "door_face": (4, 24.0, 0.85),
}
_ONE_SIDED_95_Z = 1.6448536269514722


def edge_student_gate(
    metrics: Mapping[str, float],
    *,
    profile: Mapping[str, float],
) -> dict:
    thresholds = dict(EDGE_STUDENT_GATE_THRESHOLDS)
    checks = {
        name: _metric_check(metrics, name, direction, threshold)
        for name, (direction, threshold) in thresholds.items()
    }
    subgroup = edge_subgroup_gate(metrics, profile=profile)
    checks.update(subgroup["checks"])
    thresholds.update(subgroup["thresholds"])
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
        "thresholds": {
            name: {"operator": direction, "value": threshold}
            for name, (direction, threshold) in thresholds.items()
        },
    }


def edge_subgroup_gate(
    metrics: Mapping[str, float],
    *,
    profile: Mapping[str, float],
) -> dict:
    requirements = dict(_SUBGROUP_REQUIREMENTS)
    if _profile_probability(profile, "camera_randomization") > 0.0:
        requirements["low_light"] = (1, 16.0, 0.80)
    if _profile_probability(profile, "obstacle_probability") > 0.0:
        requirements["obstacle"] = (1, 32.0, 0.85)
    checks: dict[str, bool] = {}
    thresholds: dict[str, tuple[str, float]] = {}
    episodes = metrics.get("episodes")
    episode_count = exact_episode_count(episodes)
    for prefix, (count, minimum_episodes, minimum_success) in requirements.items():
        for index in range(count):
            name = f"{prefix}_{index}" if count > 1 else prefix
            counts = subgroup_counts(
                metrics,
                prefix,
                index,
                count,
                episode_count,
            )
            valid = counts is not None
            observed, successes = counts if counts is not None else (0, 0)
            success = successes / observed if observed > 0 else 0.0
            episode_key = f"{name}_episodes"
            success_key = f"{name}_success_rate"
            thresholds[episode_key] = (">=", minimum_episodes)
            thresholds[success_key] = (">=", minimum_success)
            checks[episode_key] = valid and observed >= minimum_episodes
            checks[success_key] = valid and success >= minimum_success
    return {"checks": checks, "thresholds": thresholds}


def _metric_check(metrics, name, direction, threshold) -> bool:
    value = metrics.get(name)
    if not _valid_metric(name, value, metrics):
        return False
    episodes = metrics.get("episodes")
    episode_count = exact_episode_count(episodes)
    if name in {"success_rate", "collision_rate"}:
        count = native_fraction_count(value, episode_count)
        if count is None:
            return False
        parsed = count / episode_count
    elif name == "outside_fov_episodes":
        count = native_fraction_count(
            metrics.get("outside_fov_episode_fraction"),
            episode_count,
        )
        fraction = metrics.get("outside_fov_episode_fraction")
        if count is None or not isclose(
            float(value), float(fraction) * episode_count, rel_tol=1e-9
        ):
            return False
        parsed = float(count)
    elif name == "outside_fov_success_rate":
        outside = native_fraction_count(
            metrics.get("outside_fov_episode_fraction"),
            episode_count,
        )
        successes = native_fraction_count(
            metrics.get("outside_fov_success_fraction"),
            episode_count,
        )
        if outside is None or successes is None or successes > outside:
            return False
        parsed = successes / outside if outside > 0 else 0.0
        outside_fraction = float(metrics["outside_fov_episode_fraction"])
        reported = (
            float(metrics["outside_fov_success_fraction"]) / outside_fraction
            if outside_fraction > 0.0
            else 0.0
        )
        if not isclose(float(value), reported, rel_tol=1e-12, abs_tol=1e-12):
            return False
    else:
        parsed = float(value)
    if name == "collision_rate_upper_95":
        try:
            expected = collision_rate_upper_95(
                metrics.get("collision_rate"),
                metrics.get("episodes"),
            )
        except ValueError:
            return False
        if not isclose(float(value), expected, rel_tol=1e-12, abs_tol=1e-12):
            return False
    return parsed >= threshold if direction == ">=" else parsed <= threshold


def collision_rate_upper_95(collision_rate: object, episodes: object) -> float:
    sample_count = exact_episode_count(episodes)
    collisions = native_fraction_count(collision_rate, sample_count)
    if sample_count is None or collisions is None:
        raise ValueError("edge collision evidence requires an exact episode count")
    probability = collisions / sample_count
    z_squared = _ONE_SIDED_95_Z**2
    denominator = 1.0 + z_squared / sample_count
    center = probability + z_squared / (2.0 * sample_count)
    margin = _ONE_SIDED_95_Z * sqrt(
        probability * (1.0 - probability) / sample_count
        + z_squared / (4.0 * sample_count**2)
    )
    return min(1.0, (center + margin) / denominator)


def _valid_metric(name: str, value: object, metrics: Mapping[str, float]) -> bool:
    if not _finite_number(value):
        return False
    parsed = float(value)
    if name in _UNIT_INTERVAL_METRICS:
        return 0.0 <= parsed <= 1.0
    if name in {
        "action_rmse", "door_action_rmse", "reset_action_rmse",
        "reset_door_action_rmse", "grounding_visible_box_mae",
    }:
        return 0.0 <= parsed <= 2.0
    if name in {
        "episodes", "outside_fov_episodes", "reset_samples",
        "grounding_visible_samples", "grounding_absent_samples",
    }:
        if parsed < 0.0:
            return False
        if name in {
            "episodes", "reset_samples", "grounding_visible_samples",
            "grounding_absent_samples",
        } and not parsed.is_integer():
            return False
        if name == "outside_fov_episodes":
            episodes = metrics.get("episodes")
            return _finite_number(episodes) and parsed <= float(episodes)
        return True
    return name in {"hidden_min", "hidden_max"} and 0.0 <= parsed <= 6.0


def _profile_probability(profile: Mapping[str, float], name: str) -> float:
    value = profile.get(name)
    if not _finite_number(value) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"edge evaluation profile {name} is invalid")
    return float(value)


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(float(value))
    )
