from __future__ import annotations

from collections.abc import Mapping
from math import isclose

from flightrl.puffer4_edge_evaluation_counts import (
    exact_episode_count,
    native_fraction_count,
    subgroup_counts,
)
from flightrl.puffer4_edge_evaluation_gate import collision_rate_upper_95


_COUNT_FRACTIONS = {
    "score",
    "success_rate",
    "collision_rate",
    "outside_fov_success_fraction",
    "outside_fov_episode_fraction",
    "outside_fov_observed_fraction",
    "observed_episode_fraction",
    "low_light_episode_fraction",
    "low_light_success_fraction",
    "obstacle_episode_fraction",
    "obstacle_success_fraction",
    *(
        f"{group}_{index}_{kind}_fraction"
        for group in ("layout_family", "door_face")
        for index in range(1, 4)
        for kind in ("episode", "success")
    ),
}


def require_evaluation_metric_consistency(
    metrics: Mapping[str, float],
    *,
    configuration: Mapping[str, float],
    steps: int,
    agents: int,
) -> None:
    samples = steps * agents
    episodes = exact_episode_count(metrics.get("episodes"))
    if episodes is None or metrics.get("n") != float(episodes) or episodes > samples:
        raise ValueError("edge evaluation episode count is inconsistent")
    decoded = {
        name: native_fraction_count(metrics.get(name), episodes)
        for name in _COUNT_FRACTIONS
    }
    if any(value is None for value in decoded.values()):
        raise ValueError("edge evaluation count fraction is not native-exact")
    if metrics["score"] != metrics["success_rate"]:
        raise ValueError("edge evaluation score and success count differ")
    _require_subset_counts(decoded)
    _require_scene_groups(metrics, episodes)
    _require_derived(metrics, decoded, episodes)
    _require_sample_counts(metrics, samples, agents, episodes)
    _require_profile(metrics, configuration, decoded, episodes)


def _require_subset_counts(decoded: Mapping[str, int]) -> None:
    pairs = (
        ("outside_fov_success_fraction", "outside_fov_episode_fraction"),
        ("outside_fov_observed_fraction", "outside_fov_episode_fraction"),
        ("low_light_success_fraction", "low_light_episode_fraction"),
        ("obstacle_success_fraction", "obstacle_episode_fraction"),
    )
    if any(decoded[success] > decoded[episodes] for success, episodes in pairs):
        raise ValueError("edge evaluation subset count is inconsistent")


def _require_scene_groups(metrics: Mapping[str, float], episodes: int) -> None:
    if metrics["scene_group_schema_version"] != 1.0:
        raise ValueError("edge evaluation scene group schema version is incompatible")
    if any(
        subgroup_counts(metrics, prefix, index, 4, episodes) is None
        for prefix in ("layout_family", "door_face")
        for index in range(4)
    ):
        raise ValueError("edge evaluation scene group counts are inconsistent")


def _require_derived(metrics, decoded, episodes) -> None:
    outside = metrics["outside_fov_episode_fraction"]
    expected_outside_rate = (
        metrics["outside_fov_success_fraction"] / outside if outside > 0.0 else 0.0
    )
    expected_collision_upper = collision_rate_upper_95(
        metrics["collision_rate"], episodes
    )
    if not (
        isclose(
            metrics["outside_fov_success_rate"],
            expected_outside_rate,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        and isclose(
            metrics["outside_fov_episodes"],
            outside * episodes,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        and isclose(
            metrics["collision_rate_upper_95"],
            expected_collision_upper,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        and 0.0 <= metrics["hidden_min"] <= metrics["hidden_max"] <= 6.0
        and decoded["outside_fov_episode_fraction"] <= episodes
    ):
        raise ValueError("edge evaluation derived metrics are inconsistent")


def _require_sample_counts(metrics, samples, agents, episodes) -> None:
    counts = (
        metrics["reset_samples"],
        metrics["grounding_visible_samples"],
        metrics["grounding_absent_samples"],
    )
    if any(not float(value).is_integer() for value in counts):
        raise ValueError("edge evaluation sample count is fractional")
    resets, visible, absent = (int(value) for value in counts)
    if not (
        agents <= resets <= min(samples, agents + episodes)
        and visible + absent == samples
    ):
        raise ValueError("edge evaluation sample count is inconsistent")


def _require_profile(metrics, configuration, decoded, episodes) -> None:
    if configuration["camera_randomization"] == 0.0 and (
        decoded["low_light_episode_fraction"] != 0
        or decoded["low_light_success_fraction"] != 0
    ):
        raise ValueError("edge evaluation low-light profile evidence is inconsistent")
    probability = configuration["obstacle_probability"]
    if probability in (0.0, 1.0) and decoded["obstacle_episode_fraction"] != (
        int(probability) * episodes
    ):
        raise ValueError("edge evaluation obstacle profile evidence is inconsistent")
    if probability == 1.0 and decoded["obstacle_success_fraction"] != decoded[
        "success_rate"
    ]:
        raise ValueError("edge evaluation obstacle success evidence is inconsistent")
