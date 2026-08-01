from __future__ import annotations

from math import isfinite, sqrt


_Z_95 = 1.959963984540054
LATENCY_WARMUP_BATCHES = 16


def wilson_interval(successes: int, total: int) -> dict[str, float] | None:
    if total < 0 or successes < 0:
        raise ValueError("Wilson counts cannot be negative")
    if successes > total:
        raise ValueError("Wilson successes cannot exceed total")
    if total == 0:
        return None
    estimate = successes / total
    z2 = _Z_95 * _Z_95
    denominator = 1.0 + z2 / total
    center = (estimate + z2 / (2.0 * total)) / denominator
    margin = (
        _Z_95
        * sqrt(
            estimate * (1.0 - estimate) / total
            + z2 / (4.0 * total * total)
        )
        / denominator
    )
    return {
        "estimate": estimate,
        "low": max(0.0, center - margin),
        "high": min(1.0, center + margin),
    }


def _integer_total(value: float, name: str) -> int:
    parsed = float(value)
    if not isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be a finite non-negative count")
    result = round(parsed)
    if abs(parsed - result) > 1.0e-6:
        raise ValueError(f"{name} must be an integer count")
    return result


def _reconstruct_count(rate: float, total: int, name: str) -> int:
    parsed = float(rate)
    if not isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be a finite rate in [0, 1]")
    raw = parsed * total
    result = round(raw)
    tolerance = max(1.0e-3, total * 2.0e-7)
    if abs(raw - result) > tolerance:
        raise ValueError(f"{name} does not reconstruct an integer count")
    return result


def episode_evidence(metrics: dict[str, float]) -> dict:
    episodes = _integer_total(metrics.get("n", 0.0), "episodes")
    successes = _reconstruct_count(
        metrics.get("success_rate", 0.0),
        episodes,
        "success_rate",
    )
    collisions = _reconstruct_count(
        metrics.get("collision_rate", 0.0),
        episodes,
        "collision_rate",
    )
    outside = _reconstruct_count(
        metrics.get("outside_fov_episode_fraction", 0.0),
        episodes,
        "outside_fov_episode_fraction",
    )
    outside_successes = _reconstruct_count(
        metrics.get("outside_fov_success_fraction", 0.0),
        episodes,
        "outside_fov_success_fraction",
    )
    if outside_successes > outside:
        raise ValueError("outside-FOV successes cannot exceed episodes")
    return {
        "source": "native_binary_totals_reconstructed_from_rate_times_n",
        "counts": {
            "episodes": episodes,
            "successes": successes,
            "collisions": collisions,
            "outside_fov_episodes": outside,
            "outside_fov_successes": outside_successes,
        },
        "wilson_95": {
            "success_rate": wilson_interval(successes, episodes),
            "collision_rate": wilson_interval(collisions, episodes),
            "outside_fov_success_rate": wilson_interval(
                outside_successes,
                outside,
            ),
        },
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower
    )


def latency_summary(samples_ns: list[int]) -> dict[str, float | int | None]:
    if not samples_ns:
        return {
            "batches": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    milliseconds = [sample / 1.0e6 for sample in samples_ns]
    return {
        "batches": len(milliseconds),
        "mean": sum(milliseconds) / len(milliseconds),
        "p50": _percentile(milliseconds, 0.50),
        "p95": _percentile(milliseconds, 0.95),
        "max": max(milliseconds),
    }


def throughput(agent_steps: int, samples_ns: list[int]) -> float | None:
    elapsed_ns = sum(samples_ns)
    return agent_steps * 1.0e9 / elapsed_ns if elapsed_ns > 0 else None


def performance_report(
    *,
    batch_agents: int,
    policy_ns: list[int],
    env_ns: list[int],
    loop_ns: list[int],
) -> dict:
    policy_warmup = min(LATENCY_WARMUP_BATCHES, len(policy_ns))
    env_warmup = min(LATENCY_WARMUP_BATCHES, len(env_ns))
    loop_warmup = min(LATENCY_WARMUP_BATCHES, len(loop_ns))
    timed_policy = policy_ns[policy_warmup:]
    timed_env = env_ns[env_warmup:]
    timed_loop = loop_ns[loop_warmup:]
    mission_agent_steps = len(env_ns) * batch_agents
    return {
        "batch_agents": batch_agents,
        "agent_steps": mission_agent_steps,
        "latency_warmup": {
            "configured_batches": LATENCY_WARMUP_BATCHES,
            "excluded_batches": loop_warmup,
            "excluded_policy_batches": policy_warmup,
            "excluded_env_batches": env_warmup,
            "mission_steps_excluded": 0,
        },
        "definitions": {
            "policy_forward_batch_ms": "policy.forward_eval only",
            "native_env_step_call_batch_ms": (
                "action copy plus blocking vec.cpu_step call"
            ),
            "closed_loop_batch_ms": (
                "finite checks, policy, action postprocess, and env step"
            ),
        },
        "policy_forward_batch_ms": latency_summary(timed_policy),
        "native_env_step_call_batch_ms": latency_summary(timed_env),
        "closed_loop_batch_ms": latency_summary(timed_loop),
        "policy_agent_steps_per_second": throughput(
            len(timed_policy) * batch_agents,
            timed_policy,
        ),
        "env_agent_steps_per_second": throughput(
            len(timed_env) * batch_agents,
            timed_env,
        ),
        "closed_loop_agent_steps_per_second": throughput(
            len(timed_loop) * batch_agents,
            timed_loop,
        ),
    }


_MARGINAL_GROUPS = (
    ("layout_family", 3),
    ("door_face", 3),
    ("low_light", 1),
    ("obstacle", 1),
)


def _group_fraction_keys(
    dimension: str, category: int, categories: int
) -> tuple[str, str]:
    stem = f"{dimension}_{category}" if categories > 1 else dimension
    return f"{stem}_episode_fraction", f"{stem}_success_fraction"


def _group_row(category: int, support: int, successes: int) -> dict:
    if successes > support:
        raise ValueError("marginal group successes exceed support")
    return {
        "category": category,
        "support": support,
        "successes": successes,
        "conditional_success_rate": (
            successes / support if support > 0 else None
        ),
    }


def marginal_group_evidence(metrics: dict[str, float]) -> dict:
    if "scene_group_schema_version" not in metrics:
        return {
            "status": "unavailable",
            "reason": "scene_group_schema_version_missing",
        }
    schema = _integer_total(
        metrics["scene_group_schema_version"], "scene group schema version"
    )
    if schema != 1:
        raise ValueError("unsupported scene group schema version")
    episodes = _integer_total(metrics.get("n", 0.0), "episodes")
    total_successes = _reconstruct_count(
        metrics.get("success_rate", 0.0),
        episodes,
        "success_rate",
    )
    dimensions: dict[str, list[dict]] = {}
    supported: list[dict] = []
    for dimension, category_count in _MARGINAL_GROUPS:
        positive = []
        for category in range(1, category_count + 1):
            episode_key, success_key = _group_fraction_keys(
                dimension,
                category,
                category_count,
            )
            if episode_key not in metrics or success_key not in metrics:
                raise ValueError(f"missing marginal group metric: {episode_key}")
            positive.append(
                _group_row(
                    category,
                    _reconstruct_count(
                        metrics[episode_key],
                        episodes,
                        episode_key,
                    ),
                    _reconstruct_count(
                        metrics[success_key],
                        episodes,
                        success_key,
                    ),
                )
            )
        zero_support = episodes - sum(row["support"] for row in positive)
        zero_successes = total_successes - sum(
            row["successes"] for row in positive
        )
        if zero_support < 0 or zero_successes < 0:
            raise ValueError("marginal categories exceed episode totals")
        rows = [_group_row(0, zero_support, zero_successes), *positive]
        dimensions[dimension] = rows
        supported.extend(
            {"dimension": dimension, **row}
            for row in rows
            if row["support"] > 0
        )
    worst = (
        min(
            supported,
            key=lambda row: (
                row["conditional_success_rate"],
                row["dimension"],
                row["category"],
            ),
        )
        if supported
        else None
    )
    return {
        "status": "available",
        "schema_version": schema,
        "scope": "marginal_not_joint",
        "dimensions": dimensions,
        "worst_marginal_group": (
            {"scope": "marginal_not_joint", **worst}
            if worst is not None
            else None
        ),
    }
