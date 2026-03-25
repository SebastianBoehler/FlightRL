from __future__ import annotations

import argparse

import numpy as np

from flightrl import load_config, make_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the native FlightRL environment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()

    env = make_env(load_config(args.config), num_envs=16, seed=123)
    obs, _ = env.reset(seed=123)
    assert obs.shape[0] == env.num_agents
    for _ in range(args.steps):
        actions = np.zeros((env.num_agents, env.single_action_space.shape[0]), dtype=np.float32)
        obs, rewards, terminals, truncations, _ = env.step(actions)
        assert obs.shape[0] == env.num_agents
        assert rewards.shape[0] == env.num_agents
        assert terminals.shape[0] == env.num_agents
        assert truncations.shape[0] == env.num_agents
    env.close()
    print("smoke_test_ok")


if __name__ == "__main__":
    main()
