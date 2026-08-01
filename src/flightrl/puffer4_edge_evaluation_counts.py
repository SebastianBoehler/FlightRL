from __future__ import annotations

from math import isfinite
import struct


def exact_episode_count(value: object) -> int | None:
    if not _finite_number(value) or float(value) <= 0.0:
        return None
    return int(value) if float(value).is_integer() else None


def native_fraction_count(value: object, episodes: int | None) -> int | None:
    if (
        episodes is None
        or not _finite_number(value)
        or not 0.0 <= float(value) <= 1.0
    ):
        return None
    candidate = round(float(value) * episodes)
    encoded = struct.unpack("!f", struct.pack("!f", candidate / episodes))[0]
    return candidate if float(value) == encoded else None


def subgroup_counts(metrics, prefix, index, count, episodes):
    if episodes is None:
        return None
    if count == 1:
        return _valid_pair(
            native_fraction_count(
                metrics.get(f"{prefix}_episode_fraction"), episodes
            ),
            native_fraction_count(
                metrics.get(f"{prefix}_success_fraction"), episodes
            ),
        )
    if index > 0:
        return _valid_pair(
            native_fraction_count(
                metrics.get(f"{prefix}_{index}_episode_fraction"), episodes
            ),
            native_fraction_count(
                metrics.get(f"{prefix}_{index}_success_fraction"), episodes
            ),
        )
    subgroup_episodes = [
        native_fraction_count(
            metrics.get(f"{prefix}_{item}_episode_fraction"), episodes
        )
        for item in range(1, count)
    ]
    successes = [
        native_fraction_count(
            metrics.get(f"{prefix}_{item}_success_fraction"), episodes
        )
        for item in range(1, count)
    ]
    overall_success = native_fraction_count(metrics.get("success_rate"), episodes)
    if overall_success is None or any(
        value is None for value in (*subgroup_episodes, *successes)
    ):
        return None
    return _valid_pair(
        episodes - sum(subgroup_episodes),
        overall_success - sum(successes),
    )


def _valid_pair(episodes, successes):
    if episodes is None or successes is None or not 0 <= successes <= episodes:
        return None
    return episodes, successes


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(float(value))
    )
