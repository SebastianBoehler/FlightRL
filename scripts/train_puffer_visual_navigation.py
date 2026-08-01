from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

import torch

from flightrl.puffer4_config import Puffer4ExportSettings
from flightrl.puffer4_vision_export import export_visual_puffer4_assets
from flightrl.puffer4_vision_policy import FlightRLVisionEncoder
from flightrl.puffer4_vision_training import (
    MAX_ACTION_LOGSTD,
    MIN_ACTION_LOGSTD,
    evaluate_visual_policy,
    train_visual_policy,
    visual_simulation_gate,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUFFER_ROOT = REPO_ROOT.parent / "PufferLib-4-flightrl"


def build_puffer_environment(puffer_root: Path, env_name: str) -> None:
    env = os.environ.copy()
    if sys.platform == "darwin":
        llvm = Path("/opt/homebrew/opt/llvm/bin")
        env.update({"CC": str(llvm / "clang"), "CXX": str(llvm / "clang++")})
    subprocess.run(
        ["bash", "build.sh", env_name, "--cpu"],
        cwd=puffer_root,
        env=env,
        check=True,
    )
    if sys.platform == "darwin":
        align_openmp_runtime(puffer_root)


def align_openmp_runtime(puffer_root: Path) -> None:
    torch_openmp = Path(torch.__file__).resolve().parent / "lib" / "libomp.dylib"
    extension = (
        puffer_root / "pufferlib" / f"_C{sysconfig.get_config_var('EXT_SUFFIX')}"
    )
    dependencies = subprocess.run(
        ["otool", "-L", str(extension)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for line in dependencies.splitlines():
        dependency = line.strip().split(" ", 1)[0]
        if dependency.endswith("/libomp.dylib") and Path(dependency) != torch_openmp:
            subprocess.run(
                [
                    "install_name_tool",
                    "-change",
                    dependency,
                    str(torch_openmp),
                    str(extension),
                ],
                check=True,
            )


def load_puffer(puffer_root: Path, env_name: str):
    sys.path.insert(0, str(puffer_root))
    from pufferlib import models, pufferl, torch_pufferl

    models.FlightRLVisionEncoder = FlightRLVisionEncoder
    old_argv = sys.argv
    try:
        sys.argv = ["train_puffer_visual_navigation"]
        args = pufferl.load_config(env_name)
    finally:
        sys.argv = old_argv
    args["world_size"] = 1
    return args, torch_pufferl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and causally evaluate the native Puffer visual-navigation policy."
    )
    parser.add_argument("--puffer-root", type=Path, default=DEFAULT_PUFFER_ROOT)
    parser.add_argument("--env-name", default="flightrl_visual_navigation_smoke")
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "puffer_visual"
    )
    parser.add_argument("--total-timesteps", type=int, default=8_388_608)
    parser.add_argument("--eval-steps", type=int, default=2_080)
    parser.add_argument("--agents", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--minibatch-size", type=int, default=4_096)
    parser.add_argument("--replay-ratio", type=float, default=4.0)
    parser.add_argument("--vision-width", type=int, default=16)
    parser.add_argument("--vision-height", type=int, default=12)
    parser.add_argument("--seed", type=int, default=726)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--action-logstd", type=float, default=-0.2)
    parser.add_argument("--bootstrap-updates", type=int, default=96)
    parser.add_argument("--reset-action-head", action="store_true")
    parser.add_argument("--obstacle-probability", type=float, default=0.75)
    parser.add_argument("--domain-randomization", type=float, default=1.0)
    parser.add_argument("--evaluation-seed", type=int, default=10_007)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--log-interval", type=int, default=16)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    batch_size = args.agents * args.horizon
    if args.minibatch_size > batch_size or args.minibatch_size % args.horizon:
        parser.error(
            "--minibatch-size must not exceed agents*horizon and must be "
            "divisible by --horizon"
        )
    if not 0.0 <= args.obstacle_probability <= 1.0:
        parser.error("--obstacle-probability must be between 0 and 1")
    if not 0.0 <= args.domain_randomization <= 1.0:
        parser.error("--domain-randomization must be between 0 and 1")
    settings = Puffer4ExportSettings(
        env_name=args.env_name,
        total_agents=args.agents,
        num_buffers=1,
        num_threads=args.threads,
        policy_hidden_size=args.hidden_size,
        policy_num_layers=1,
        train_seed=args.seed,
    )
    result = export_visual_puffer4_assets(
        args.puffer_root,
        settings,
        vision_width=args.vision_width,
        vision_height=args.vision_height,
    )
    if not args.skip_build:
        build_puffer_environment(args.puffer_root, result.env_name)
    puffer_args, torch_pufferl = load_puffer(args.puffer_root, result.env_name)
    puffer_args["train"]["learning_rate"] = args.learning_rate
    puffer_args["train"]["ent_coef"] = args.entropy_coef
    puffer_args["train"]["horizon"] = args.horizon
    puffer_args["train"]["minibatch_size"] = args.minibatch_size
    puffer_args["train"]["replay_ratio"] = args.replay_ratio
    puffer_args["env"]["obstacle_probability"] = args.obstacle_probability
    puffer_args["env"]["domain_randomization"] = args.domain_randomization
    if args.init_checkpoint:
        puffer_args["load_model_path"] = str(
            args.init_checkpoint.expanduser().resolve()
        )
    trainer, history, elapsed, selection, bootstrap = train_visual_policy(
        puffer_args,
        torch_pufferl,
        args.total_timesteps,
        args.log_interval,
        args.action_logstd,
        args.reset_action_head,
        args.bootstrap_updates,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / f"{args.env_name}_{args.total_timesteps}.bin"
    ppo_state = {
        key: value.detach().cpu().clone()
        for key, value in trainer.policy.state_dict().items()
    }
    candidate_states = {"bootstrap": bootstrap["state"], "ppo": ppo_state}
    candidate_evaluations = {}
    for label, state in candidate_states.items():
        trainer.policy.load_state_dict(state)
        candidate_evaluations[label] = {
            "obstacle_full_vision": evaluate_visual_policy(
                trainer.policy,
                puffer_args,
                torch_pufferl,
                args.eval_steps,
                "full",
                1.0,
                seed=args.evaluation_seed,
            ),
            "obstacle_masked_vision": evaluate_visual_policy(
                trainer.policy,
                puffer_args,
                torch_pufferl,
                args.eval_steps,
                "masked",
                1.0,
                seed=args.evaluation_seed,
            ),
            "clear_full_vision": evaluate_visual_policy(
                trainer.policy,
                puffer_args,
                torch_pufferl,
                args.eval_steps,
                "full",
                0.0,
                seed=args.evaluation_seed,
            ),
            "nominal_obstacle_full_vision": evaluate_visual_policy(
                trainer.policy,
                puffer_args,
                torch_pufferl,
                args.eval_steps,
                "full",
                1.0,
                domain_randomization=0.0,
                seed=args.evaluation_seed,
            ),
        }
    selected_label = max(
        candidate_evaluations,
        key=lambda label: candidate_score(candidate_evaluations[label]),
    )
    trainer.policy.load_state_dict(candidate_states[selected_label])
    trainer.save_weights(checkpoint)
    evaluation = candidate_evaluations[selected_label]
    simulation_gate = visual_simulation_gate(evaluation)
    trainer.close()
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "trainer": "PufferLib PuffeRL PPO with MinGRU",
        "learning_rate": args.learning_rate,
        "entropy_coef": args.entropy_coef,
        "initial_action_logstd": args.action_logstd,
        "action_logstd_bounds": [MIN_ACTION_LOGSTD, MAX_ACTION_LOGSTD],
        "bootstrap": bootstrap["bootstrap"],
        "reset_action_head": args.reset_action_head,
        "horizon": args.horizon,
        "minibatch_size": args.minibatch_size,
        "replay_ratio": args.replay_ratio,
        "training_obstacle_probability": args.obstacle_probability,
        "domain_randomization": args.domain_randomization,
        "navigation_residual_scale": puffer_args["env"][
            "navigation_residual_scale"
        ],
        "evaluation_seed": args.evaluation_seed,
        "policy_parameters": trainer.model_size,
        "observation": (
            f"3x{args.vision_height}x{args.vision_width} gray4 "
            "appearance/delta/motion plus 6-value goal intent"
        ),
        "vision_width": args.vision_width,
        "vision_height": args.vision_height,
        "action": "learned residual over body vx/vy, world vz, and yaw-rate waypoint tracking",
        "total_timesteps": args.total_timesteps,
        "elapsed_s": elapsed,
        "history": history,
        "training_selection": selection,
        "candidate_selection": selected_label,
        "candidate_evaluations": candidate_evaluations,
        "evaluation": evaluation,
        "camera_is_causal": evaluation["obstacle_full_vision"].get(
            "success_rate", 0.0
        )
        > evaluation["obstacle_masked_vision"].get("success_rate", 0.0),
        "simulation_gate": simulation_gate,
        "deployment_status": (
            "simulation gate passed; shadow replay required before live control"
            if simulation_gate["passed"]
            else "simulation gate failed; not approved for live control"
        ),
    }
    report_path = checkpoint.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"checkpoint={checkpoint}")
    print(f"report={report_path}")
    print(f"evaluation={report['evaluation']}")


def candidate_score(evaluation: dict) -> float:
    obstacle = evaluation["obstacle_full_vision"]
    masked = evaluation["obstacle_masked_vision"]
    clear = evaluation["clear_full_vision"]
    return (
        3.0 * obstacle.get("success_rate", 0.0)
        - 3.0 * obstacle.get("collision_rate", 1.0)
        + clear.get("success_rate", 0.0)
        - masked.get("success_rate", 0.0)
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
