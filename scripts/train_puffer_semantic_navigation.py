from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

from flightrl.mujoco.semantic_imitation import bootstrap_imitation
from flightrl.mujoco.semantic_puffer_config import semantic_puffer_config
from flightrl.mujoco.semantic_puffer_driver import SemanticPufferDriver
from flightrl.mujoco.semantic_puffer_trainer import StatefulSemanticPuffeRL
from flightrl.mujoco.semantic_reporting import (
    build_semantic_training_report,
    select_semantic_candidate,
)
from flightrl.mujoco.semantic_run import (
    SemanticDriverConfig,
    evaluate_policy_rooms,
    evaluate_teacher_rooms,
)
from flightrl.mujoco.semantic_vision_policy import SemanticVisionPolicy


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUFFER_ROOT = REPO_ROOT.parent / "PufferLib-4-flightrl"


def train_ppo(
    policy: SemanticVisionPolicy,
    driver: SemanticPufferDriver,
    args: dict,
    torch_pufferl,
) -> tuple[object, list[dict], float]:
    base_trainer = torch_pufferl.PuffeRL(args, driver, policy, verbose=False)
    trainer = StatefulSemanticPuffeRL(base_trainer, torch_pufferl)
    epochs = max(1, args["train"]["total_timesteps"] // trainer.batch_size)
    history = []
    pending_env = []
    started = perf_counter()
    for epoch in range(1, epochs + 1):
        trainer.rollouts()
        if trainer.env_logs:
            pending_env.append(trainer.env_logs)
        trainer.train()
        if epoch == 1 or epoch == epochs or epoch % 32 == 0:
            logs = trainer.log()
            env_logs = aggregate_logs(pending_env)
            pending_env.clear()
            entry = {
                "epoch": epoch,
                "agent_steps": trainer.global_step,
                "sps": logs["SPS"],
                "env": env_logs,
                "loss": logs["loss"],
            }
            history.append(entry)
            print(
                f"ppo={epoch}/{epochs} steps={trainer.global_step} "
                f"sps={logs['SPS']:.0f} env={env_logs}",
                flush=True,
            )
    return trainer, history, perf_counter() - started


def aggregate_logs(logs: list[dict]) -> dict:
    count = sum(entry.get("n", 0.0) for entry in logs)
    if count == 0:
        return {}
    keys = set().union(*(entry.keys() for entry in logs)) - {"n"}
    result = {
        key: sum(entry.get(key, 0.0) * entry.get("n", 0.0) for entry in logs) / count
        for key in keys
    }
    result["n"] = count
    return result


def load_puffer(puffer_root: Path):
    sys.path.insert(0, str(puffer_root))
    import pufferlib
    from pufferlib import torch_pufferl

    if pufferlib.__version__ != 4.0:
        raise RuntimeError(f"expected PufferLib 4.0, got {pufferlib.__version__}")
    return torch_pufferl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap and train recurrent semantic navigation with PufferLib 4"
    )
    parser.add_argument("--puffer-root", type=Path, default=DEFAULT_PUFFER_ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "artifacts/puffer_semantic"
    )
    parser.add_argument("--run-tag", default="v2")
    parser.add_argument("--seed", type=int, default=726)
    parser.add_argument("--rooms", type=int, default=4)
    parser.add_argument("--agents-per-room", type=int, default=2)
    parser.add_argument("--eval-rooms", type=int, default=4)
    parser.add_argument("--eval-agents-per-room", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--vision-width", type=int, default=64)
    parser.add_argument("--vision-height", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--bootstrap-updates", type=int, default=64)
    parser.add_argument("--bootstrap-learning-rate", type=float, default=0.001)
    parser.add_argument("--clearance-loss-scale", type=float, default=1.0)
    parser.add_argument("--collision-risk-loss-scale", type=float, default=2.0)
    parser.add_argument("--total-timesteps", type=int, default=32_768)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--eval-steps", type=int, default=800)
    parser.add_argument(
        "--ablation-steps",
        type=int,
        default=0,
        help="Steps for non-causal ablations; zero uses --eval-steps",
    )
    parser.add_argument("--active-exploration", action="store_true")
    parser.add_argument(
        "--room-profile",
        choices=("standard", "diverse"),
        default="standard",
    )
    parser.add_argument(
        "--eval-room-profile",
        choices=("standard", "diverse"),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch_pufferl = load_puffer(args.puffer_root.resolve())
    train_config = SemanticDriverConfig(
        seed=args.seed,
        room_count=args.rooms,
        agents_per_room=args.agents_per_room,
        active_exploration=args.active_exploration,
        vision_width=args.vision_width,
        vision_height=args.vision_height,
        room_profile=args.room_profile,
    )
    eval_config = SemanticDriverConfig(
        seed=args.seed,
        room_count=args.eval_rooms,
        agents_per_room=args.eval_agents_per_room,
        offset=10_000,
        active_exploration=args.active_exploration,
        vision_width=args.vision_width,
        vision_height=args.vision_height,
        room_profile=args.eval_room_profile or args.room_profile,
    )
    train_driver = train_config.build()
    policy = SemanticVisionPolicy(
        train_driver.driver_env,
        hidden_size=args.hidden_size,
        recurrent_safety=False,
        recurrent_visual_safety=True,
    )
    with torch.no_grad():
        if args.active_exploration:
            policy.decoder.decoder_logstd.fill_(-3.0)
            policy.decoder.decoder_logstd[:, 1:3].fill_(-9.0)
        else:
            policy.decoder.decoder_logstd[:, :3].fill_(-4.0)
            policy.decoder.decoder_logstd[:, 3:].fill_(-2.2)
    started = perf_counter()
    imitation = bootstrap_imitation(
        policy,
        train_driver,
        updates=args.bootstrap_updates,
        horizon=args.horizon,
        learning_rate=args.bootstrap_learning_rate,
        clearance_loss_scale=args.clearance_loss_scale,
        collision_risk_loss_scale=args.collision_risk_loss_scale,
    )
    policy.freeze_visual_safety_encoder()
    bootstrap_state = deepcopy(policy.state_dict())
    config = semantic_puffer_config(
        args.total_timesteps,
        train_driver.total_agents,
        args.horizon,
        args.learning_rate,
    )
    trainer, history, ppo_elapsed = train_ppo(
        policy,
        train_driver,
        config,
        torch_pufferl,
    )
    final_state = deepcopy(trainer.policy.state_dict())
    trainer.close()

    evaluations = {}
    candidates = {"bootstrap": bootstrap_state, "puffer_ppo": final_state}
    for candidate, state in candidates.items():
        policy.load_state_dict(state)
        full = evaluate_policy_rooms(
            policy,
            eval_config,
            mode="full",
            steps=args.eval_steps,
        )
        print(f"eval mode=full {full}")
        evaluations[candidate] = {
            "full": full,
        }

    teacher_evaluation = evaluate_teacher_rooms(
        eval_config,
        steps=args.eval_steps,
    )
    print(f"eval teacher {teacher_evaluation}")
    selected = select_semantic_candidate(
        evaluations,
        active_exploration=args.active_exploration,
    )
    policy.load_state_dict(candidates[selected])
    ablation_modes = ["vision_masked", "target_map_masked", "command_rotated"]
    if args.active_exploration:
        ablation_modes.append("temporal_masked")
    for mode in ablation_modes:
        steps = (
            args.eval_steps
            if mode == "target_map_masked"
            else args.ablation_steps or args.eval_steps
        )
        result = evaluate_policy_rooms(
            policy,
            eval_config,
            mode=mode,
            steps=steps,
        )
        print(f"eval mode={mode} {result}")
        evaluations[selected][mode] = result
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = (
        args.output_dir
        / f"semantic_nav_{args.run_tag}_seed{args.seed}_{args.total_timesteps}.pt"
    )
    candidate_checkpoints = {}
    for candidate, state in candidates.items():
        candidate_path = checkpoint.with_name(
            f"{checkpoint.stem}.{candidate}{checkpoint.suffix}"
        )
        torch.save(state, candidate_path)
        candidate_checkpoints[candidate] = {
            "checkpoint": str(candidate_path),
            "sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        }
    torch.save(policy.state_dict(), checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = build_semantic_training_report(
        checkpoint=checkpoint,
        digest=digest,
        run_tag=args.run_tag,
        selected=selected,
        policy=policy,
        seed=args.seed,
        rooms=args.rooms,
        eval_rooms=args.eval_rooms,
        bootstrap_updates=args.bootstrap_updates,
        imitation=imitation,
        puffer_timesteps=args.total_timesteps,
        puffer_elapsed_s=ppo_elapsed,
        total_elapsed_s=perf_counter() - started,
        max_horizontal_speed_m_s=(
            train_driver.driver_env.control.max_horizontal_speed_m_s
        ),
        history=history,
        evaluations=evaluations,
        teacher_evaluation=teacher_evaluation,
        active_exploration=args.active_exploration,
        room_profile=args.room_profile,
        eval_room_profile=args.eval_room_profile or args.room_profile,
        clearance_loss_scale=args.clearance_loss_scale,
        collision_risk_loss_scale=args.collision_risk_loss_scale,
    )
    report["candidate_checkpoints"] = candidate_checkpoints
    report_path = checkpoint.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"selected={selected}")
    print(f"checkpoint={checkpoint}")
    print(f"report={report_path}")
    print(f"sha256={digest}")
if __name__ == "__main__":
    main()
