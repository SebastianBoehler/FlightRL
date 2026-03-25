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
    assert "x" in env.snapshot(0)
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
