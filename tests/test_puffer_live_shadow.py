from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evaluate_puffer_sixdof_live_log", ROOT / "scripts" / "evaluate_puffer_sixdof_live_log.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
build_report = MODULE.build_report


@dataclass(frozen=True)
class Metadata:
    observation_dim: int = 28
    hidden_size: int = 16
    action_dim: int = 4
    num_layers: int = 2


def test_puffer_live_shadow_report_groups_vertical_clearance() -> None:
    args = SimpleNamespace(checkpoint="checkpoint.bin", input="live.csv", task="obstacle_avoidance")
    policy = SimpleNamespace(metadata=Metadata())
    pairs = [
        (
            np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.asarray([-0.4, 0.0, 0.0, 0.0], dtype=np.float32),
            row(range_up=200.0, range_zrange=500.0),
        ),
        (
            np.asarray([0.2, 0.0, 0.0, 0.0], dtype=np.float32),
            np.asarray([0.4, 0.0, 0.0, 0.0], dtype=np.float32),
            row(range_up=2000.0, range_zrange=180.0),
        ),
    ]

    report = build_report(args, policy, pairs)

    assert report["groups"]["top_lt_45cm"]["samples"] == 1
    assert report["groups"]["bottom_lt_25cm"]["samples"] == 1
    assert report["groups"]["vertical_lt_35cm"]["samples"] == 2
    assert report["groups"]["vertical_lt_35cm"]["min_top_range_m"] == 0.2
    assert report["groups"]["vertical_lt_35cm"]["min_bottom_range_m"] == 0.18
    assert "thrust" in report["groups"]["all"]["sign_agreement"]
    assert MODULE.live_range_m({"range.up": 0.0}, "range.up") == 4.0


def row(*, range_up: float, range_zrange: float) -> dict[str, float]:
    return {
        "range.front": 2000.0,
        "range.back": 2000.0,
        "range.left": 2000.0,
        "range.right": 2000.0,
        "range.up": range_up,
        "range.zrange": range_zrange,
        "min_horizontal_ttc_s": 9.0,
    }
