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
