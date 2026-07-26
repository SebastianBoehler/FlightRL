from __future__ import annotations

from pathlib import Path

from flightrl import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_load_hover_config() -> None:
    config = load_config(ROOT / "configs" / "tasks" / "hover.toml")
    assert config.environment.action_mode == "stabilized_planar"
    assert config.action_dim == 2
    assert config.observation_dim > 0


def test_motor_quad_config_changes_action_dim_and_observation_dim() -> None:
    config = load_config(
        ROOT / "configs" / "tasks" / "hover.toml",
        overrides={"environment": {"action_mode": "motor_quad"}},
    )
    assert config.action_dim == 4
    assert config.observation_dim > 0


def test_hover_command_config_matches_live_command_shape() -> None:
    config = load_config(ROOT / "configs" / "tasks" / "crazyflie_hover_command.toml")
    assert config.environment.action_mode == "hover_command"
    assert config.action_dim == 4
    assert config.observation_dim > 0


def test_vision_sensor_reserves_configurable_observation_slots() -> None:
    baseline = load_config(ROOT / "configs" / "tasks" / "hover.toml")
    config = load_config(
        ROOT / "configs" / "tasks" / "hover.toml",
        overrides={
            "sensors": {"include_vision_sensor": True},
            "vision": {
                "width": 8,
                "height": 6,
                "frame_stack": 2,
                "include_delta": True,
                "include_motion_mask": True,
            },
        },
    )

    assert config.vision.shape == (4, 6, 8)
    assert config.observation_dim == baseline.observation_dim + 4 * 6 * 8
    assert config.vision_slice == slice(baseline.observation_dim, config.observation_dim)
