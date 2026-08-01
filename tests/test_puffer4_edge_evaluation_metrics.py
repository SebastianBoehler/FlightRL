from __future__ import annotations

import pytest

from flightrl.puffer4_edge_evaluation_gate import EDGE_EVALUATION_PROFILES
from flightrl.puffer4_edge_evaluation_metrics import (
    require_evaluation_metric_consistency,
)
from puffer4_edge_shadow_support import passing_metrics


STEPS = 6000
AGENTS = 128
PROFILE = EDGE_EVALUATION_PROFILES[1][3]


def _require(metrics: dict[str, float]) -> None:
    require_evaluation_metric_consistency(
        metrics,
        configuration=PROFILE,
        steps=STEPS,
        agents=AGENTS,
    )


def test_evaluation_metrics_require_native_exact_bounded_counts() -> None:
    _require(passing_metrics(PROFILE))

    metrics = passing_metrics(PROFILE)
    metrics["episodes"] = float(STEPS * AGENTS + 1)
    metrics["n"] = metrics["episodes"]
    with pytest.raises(ValueError, match="episode count"):
        _require(metrics)


def test_evaluation_metrics_reject_non_native_count_fractions() -> None:
    metrics = passing_metrics(PROFILE)
    metrics["low_light_episode_fraction"] = 64.1 / metrics["episodes"]

    with pytest.raises(ValueError, match="native-exact"):
        _require(metrics)


def test_evaluation_metrics_bound_reset_and_grounding_exposure() -> None:
    metrics = passing_metrics(PROFILE)
    metrics["reset_samples"] = 385.0
    with pytest.raises(ValueError, match="sample count"):
        _require(metrics)

    metrics = passing_metrics(PROFILE)
    metrics["grounding_absent_samples"] -= 1.0
    with pytest.raises(ValueError, match="sample count"):
        _require(metrics)
