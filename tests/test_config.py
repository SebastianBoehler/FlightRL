from __future__ import annotations

from pathlib import Path

import pytest

from flightrl import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_load_hover_config() -> None:
    config = load_config(ROOT / "configs" / "tasks" / "hover.toml")
    assert config.environment.action_mode == "stabilized_planar"
    assert config.observation_dim > 0


def test_range_sensor_placeholder_fails_fast() -> None:
    with pytest.raises(NotImplementedError):
        load_config(
            ROOT / "configs" / "tasks" / "hover.toml",
            overrides={"sensors": {"include_range_sensor": True}},
        )
