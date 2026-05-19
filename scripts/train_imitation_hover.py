from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from flightrl import load_config, make_env
from flightrl.training import create_env_and_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a FlightRL hover policy from a deterministic teacher")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/crazyflie_hover_imitation.pt")
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--steps-per-update", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--policy-hidden-size", type=int, default=None)
    parser.add_argument("--policy-num-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = load_config(args.config, overrides={"environment": {"num_envs": args.num_envs}})
    env, policy = create_env_and_policy(
        config,
        seed=args.seed,
        policy_hidden_size=args.policy_hidden_size,
        policy_num_layers=args.policy_num_layers,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    obs, _ = env.reset(seed=args.seed)

    for update in range(1, args.updates + 1):
        obs_batch: list[np.ndarray] = []
        act_batch: list[np.ndarray] = []
        for _ in range(args.steps_per_update):
            actions = teacher_actions(env)
            obs_batch.append(obs.copy())
            act_batch.append(actions.copy())
            obs, _rewards, terminals, truncations, _info = env.step(actions)
            if np.any(terminals) or np.any(truncations):
                obs, _ = env.reset()

        observations = torch.from_numpy(np.concatenate(obs_batch, axis=0)).float()
        targets = torch.from_numpy(np.concatenate(act_batch, axis=0)).float()
        permutation = torch.randperm(observations.shape[0])
        losses = []
        for start in range(0, observations.shape[0], args.batch_size):
            indices = permutation[start : start + args.batch_size]
            logits, _values, _state = policy.forward_eval(observations[indices])
            loss = F.mse_loss(logits.mean, targets[indices])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        if update == 1 or update % max(1, args.updates // 10) == 0:
            print(f"update={update} loss={np.mean(losses):.6f}")

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), checkpoint)
    env.close()
    print(f"checkpoint={checkpoint}")


def teacher_actions(env) -> np.ndarray:
    config = env.config
    actions = np.zeros((env.num_agents, config.action_dim), dtype=np.float32)
    for index in range(env.num_agents):
        snapshot = env.snapshot(index)
        target_z = snapshot["target_z"]
        target_x = snapshot["target_x"]
        z_error = target_z - snapshot["z"]
        x_error = target_x - snapshot["x"]
        pitch_target = np.clip(-0.7 * x_error - 0.18 * snapshot["vx"], -0.35, 0.35)
        thrust = 0.75 * z_error - 0.18 * snapshot["vz"]
        pitch = 2.2 * (pitch_target - snapshot["pitch"]) - 0.25 * snapshot["pitch_rate"]
        actions[index, 0] = np.clip(thrust, -1.0, 1.0)
        actions[index, 1] = np.clip(pitch, -1.0, 1.0)
    return actions


if __name__ == "__main__":
    main()
