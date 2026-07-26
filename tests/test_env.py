from __future__ import annotations

from pathlib import Path

import numpy as np

from flightrl import load_config, make_env


ROOT = Path(__file__).resolve().parents[1]


def test_reset_and_step_shapes() -> None:
    config = load_config(ROOT / "configs" / "tasks" / "hover.toml", overrides={"environment": {"num_envs": 4}})
    env = make_env(config, seed=7)
    obs, _ = env.reset(seed=7)
    assert obs.shape == (4, config.observation_dim)

    actions = np.zeros((4, config.action_dim), dtype=np.float32)
    next_obs, rewards, terminals, truncations, _ = env.step(actions)
    assert next_obs.shape == obs.shape
    assert rewards.shape == (4,)
    assert terminals.shape == (4,)
    assert truncations.shape == (4,)
    snapshot = env.snapshot(0)
    assert "x" in snapshot
    assert "motor_front_left" in snapshot
    assert "command_0" in snapshot
    env.close()


def test_host_fed_vision_observation_persists_across_native_step() -> None:
    config = load_config(
        ROOT / "configs" / "tasks" / "hover.toml",
        overrides={
            "environment": {"num_envs": 2},
            "sensors": {"include_vision_sensor": True},
            "vision": {"width": 2, "height": 2, "normalization": "zero_one"},
        },
    )
    env = make_env(config, seed=9)
    env.reset(seed=9)
    with np.testing.assert_raises_regex(RuntimeError, "set_vision"):
        env.step(np.zeros((2, config.action_dim), dtype=np.float32))

    encoded = env.set_vision_frames(
        (
            np.zeros((2, 2), dtype=np.uint8),
            np.full((2, 2), 255, dtype=np.uint8),
        )
    )
    assert encoded.shape == (2, 1, 2, 2)
    assert np.all(env.observations[0, config.vision_slice] == 0.0)
    assert np.all(env.observations[1, config.vision_slice] == 1.0)

    env.step(np.zeros((2, config.action_dim), dtype=np.float32))
    assert np.all(env.observations[1, config.vision_slice] == 1.0)

    observations, _ = env.reset(seed=9)
    env.close()
    assert np.all(observations[:, config.vision_slice] == 0.0)


def test_deterministic_rollout_is_reproducible() -> None:
    config = load_config(
        ROOT / "configs" / "tasks" / "sequence.toml",
        overrides={"environment": {"num_envs": 1}, "task": {"max_steps": 12}},
    )
    action = np.array([[0.0, 0.0]], dtype=np.float32)
    traces = []
    for _ in range(2):
        env = make_env(config, seed=11)
        env.reset(seed=11)
        rollout = []
        for _ in range(6):
            obs, rewards, terminals, truncations, _ = env.step(action)
            rollout.append((obs.copy(), rewards.copy(), terminals.copy(), truncations.copy()))
        traces.append(rollout)
        env.close()

    for left, right in zip(traces[0], traces[1], strict=True):
        for lhs, rhs in zip(left, right, strict=True):
            assert np.allclose(lhs, rhs)


def test_timeout_sets_truncation() -> None:
    config = load_config(
        ROOT / "configs" / "tasks" / "hover.toml",
        overrides={"environment": {"num_envs": 1}, "task": {"max_steps": 2}},
    )
    env = make_env(config, seed=3)
    env.reset(seed=3)
    action = np.zeros((1, config.action_dim), dtype=np.float32)
    truncated = False
    for _ in range(4):
        _, _, _, truncations, _ = env.step(action)
        if truncations[0]:
            truncated = True
            break
    env.close()
    assert truncated


def test_rgb_array_render_returns_frame() -> None:
    config = load_config(
        ROOT / "configs" / "tasks" / "hover.toml",
        overrides={"environment": {"num_envs": 1}},
    )
    env = make_env(config, seed=5, render_mode="rgb_array")
    env.reset(seed=5)
    frame = env.render()
    env.close()
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.dtype == np.uint8


def test_motor_quad_mode_uses_four_actions_and_changes_pitch() -> None:
    config = load_config(
        ROOT / "configs" / "tasks" / "hover.toml",
        overrides={"environment": {"num_envs": 1, "action_mode": "motor_quad"}},
    )
    assert config.action_dim == 4
    env = make_env(config, seed=13)
    env.reset(seed=13)
    action = np.array([[1.0, 1.0, -1.0, -1.0]], dtype=np.float32)
    pitch_before = env.snapshot(0)["pitch"]
    for _ in range(5):
        env.step(action)
    pitch_after = env.snapshot(0)["pitch"]
    env.close()
    assert abs(pitch_after - pitch_before) > 1e-5


def test_hover_command_mode_uses_live_command_shape() -> None:
    config = load_config(
        ROOT / "configs" / "tasks" / "crazyflie_hover_command.toml",
        overrides={"environment": {"num_envs": 1, "reset_mode": "deterministic"}, "task": {"fixed_start": [0.0, 0.35], "target_bounds": [-0.2, 0.2, 0.35, 0.65]}},
    )
    assert config.action_dim == 4
    env = make_env(config, seed=17)
    env.reset(seed=17)
    action = np.array([[1.0, 0.5, -0.5, 1.0]], dtype=np.float32)
    before = env.snapshot(0)
    for _ in range(12):
        env.step(action)
    after = env.snapshot(0)
    env.close()

    assert after["command_0"] == 1.0
    assert after["command_1"] == 0.5
    assert after["command_2"] == -0.5
    assert after["command_3"] == 1.0
    assert after["x"] > before["x"]
    assert after["z"] > before["z"]


def test_wind_changes_trajectory_deterministically() -> None:
    overrides = {
        "environment": {"num_envs": 1},
        "wind": {"enabled": True, "steady_x": 2.0, "steady_z": 0.0, "gust_strength": 0.4, "gust_tau": 0.3},
    }
    config = load_config(ROOT / "configs" / "tasks" / "hover.toml", overrides=overrides)
    action = np.zeros((1, config.action_dim), dtype=np.float32)
    traces = []
    for seed in (19, 19):
        env = make_env(config, seed=seed)
        env.reset(seed=seed)
        rollout = []
        for _ in range(6):
            env.step(action)
            snapshot = env.snapshot(0)
            rollout.append((snapshot["x"], snapshot["z"], snapshot["wind_x"], snapshot["wind_z"]))
        traces.append(rollout)
        env.close()
    assert traces[0] == traces[1]
    assert any(abs(x) > 1e-5 for x, _, _, _ in traces[0])
