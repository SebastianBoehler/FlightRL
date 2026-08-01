from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from flightrl.puffer4_config import Puffer4ExportSettings
from flightrl.puffer4_door_contract import CORRECTED_DOOR_ACTION_CONTRACT
from flightrl.puffer4_door_export import export_fixed_door_assets
from flightrl.puffer4_door_evaluation import evaluate_door_candidates
from flightrl.puffer4_door_policy_contract import door_policy_contract_report
from flightrl.puffer4_door_stream_contract import door_stream_contract_report
from flightrl.puffer4_door_reporting import persist_grounder_failure
from flightrl.puffer4_door_runner import (
    accepted_observability,
    build_environment,
    load_puffer,
)
from flightrl.puffer4_door_training import (
    evaluate_door_teacher,
    fixed_door_teacher_gate,
    train_door_policy,
)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUFFER = ROOT.parent / "PufferLib-4-flightrl"
DEFAULT_OBSERVABILITY = (
    ROOT
    / "artifacts/semantic/door-observability-64x48-r128-20260729"
    / "door_observability.pt"
)
DEFAULT_REAL_GATE = (
    ROOT
    / "artifacts/semantic/door-observability-real-gate-20260729"
    / "report.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the no-oracle recurrent fixed-door D1 Puffer policy."
    )
    parser.add_argument("--puffer-root", type=Path, default=DEFAULT_PUFFER)
    parser.add_argument("--env-name", default="flightrl_fixed_door_d1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/puffer_fixed_door",
    )
    parser.add_argument(
        "--observability-checkpoint",
        type=Path,
        default=DEFAULT_OBSERVABILITY,
    )
    parser.add_argument("--real-gate-report", type=Path, default=DEFAULT_REAL_GATE)
    parser.add_argument("--total-timesteps", type=int, default=8_388_608)
    parser.add_argument("--eval-steps", type=int, default=6_000)
    parser.add_argument("--screen-steps", type=int, default=600)
    parser.add_argument("--eval-agents", type=int, default=128)
    parser.add_argument("--agents", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--replay-ratio", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--entropy-coef", type=float, default=0.002)
    parser.add_argument("--obstacle-probability", type=float, default=0.5)
    parser.add_argument("--expanded-layouts", action="store_true")
    parser.add_argument(
        "--hardware-camera-randomization",
        action="store_true",
    )
    parser.add_argument("--bootstrap-updates", type=int, default=96)
    parser.add_argument("--bootstrap-learning-rate", type=float, default=0.001)
    parser.add_argument("--bootstrap-max-policy-rollin", type=float, default=0.2)
    parser.add_argument("--grounder-updates", type=int, default=512)
    parser.add_argument("--grounder-learning-rate", type=float, default=0.002)
    parser.add_argument("--grounder-eval-batches", type=int, default=64)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--evaluation-seed", type=int, default=10_011)
    parser.add_argument("--log-interval", type=int, default=16)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.obstacle_probability <= 1.0:
        parser.error("--obstacle-probability must be between 0 and 1")
    if not 0.0 <= args.bootstrap_max_policy_rollin <= 1.0:
        parser.error("--bootstrap-max-policy-rollin must be between 0 and 1")

    batch_size = args.agents * args.horizon
    if args.minibatch_size > batch_size or args.minibatch_size % args.horizon:
        parser.error(
            "--minibatch-size must not exceed agents*horizon and must be "
            "divisible by --horizon"
        )
    observability = accepted_observability(
        root=ROOT,
        checkpoint=args.observability_checkpoint,
        gate_report=args.real_gate_report,
    )
    settings = Puffer4ExportSettings(
        env_name=args.env_name,
        total_agents=args.agents,
        num_buffers=1,
        num_threads=args.threads,
        policy_hidden_size=args.hidden_size,
        policy_num_layers=1,
        train_seed=args.seed,
    )
    export_fixed_door_assets(args.puffer_root, settings)
    if not args.skip_build:
        build_environment(args.puffer_root, args.env_name)
    puffer_args, torch_pufferl = load_puffer(args.puffer_root, args.env_name)
    puffer_args["train"].update(
        {
            "learning_rate": args.learning_rate,
            "ent_coef": args.entropy_coef,
            "horizon": args.horizon,
            "minibatch_size": args.minibatch_size,
            "replay_ratio": args.replay_ratio,
        }
    )
    puffer_args["env"]["obstacle_probability"] = args.obstacle_probability
    puffer_args["env"]["layout_diversity"] = float(args.expanded_layouts)
    puffer_args["env"]["camera_randomization"] = float(
        args.hardware_camera_randomization
    )
    CORRECTED_DOOR_ACTION_CONTRACT.verify_env(puffer_args["env"])
    teacher = evaluate_door_teacher(
        puffer_args,
        torch_pufferl,
        steps=args.eval_steps,
        seed=args.evaluation_seed,
        agents=args.eval_agents,
    )
    teacher_gate = fixed_door_teacher_gate(teacher)
    print(f"teacher={teacher} gate={teacher_gate}", flush=True)
    if not teacher_gate["passed"]:
        raise RuntimeError(
            f"fixed-door teacher gate failed: {teacher_gate['failures']}"
        )
    trainer, history, elapsed, bootstrap, bootstrap_state = train_door_policy(
        puffer_args,
        torch_pufferl,
        observability_checkpoint=observability,
        total_timesteps=args.total_timesteps,
        bootstrap_updates=args.bootstrap_updates,
        bootstrap_learning_rate=args.bootstrap_learning_rate,
        bootstrap_max_policy_rollin=args.bootstrap_max_policy_rollin,
        log_interval=args.log_interval,
        initial_policy_state=(
            torch.load(args.initial_checkpoint, map_location="cpu", weights_only=True)
            if args.initial_checkpoint is not None
            else None
        ),
        grounder_updates=args.grounder_updates,
        grounder_learning_rate=args.grounder_learning_rate,
        grounder_eval_batches=args.grounder_eval_batches,
        grounder_evaluation_seed=args.evaluation_seed,
    )
    if trainer is None:
        checkpoint, report_path = persist_grounder_failure(
            output_dir=args.output_dir,
            env_name=args.env_name,
            seed=args.seed,
            state=bootstrap_state,
            bootstrap=bootstrap,
            metadata={
                "grounder_updates": args.grounder_updates,
                "grounder_learning_rate": args.grounder_learning_rate,
                "training_obstacle_probability": args.obstacle_probability,
                "expanded_layouts": args.expanded_layouts,
                "hardware_camera_randomization": (
                    args.hardware_camera_randomization
                ),
                "evaluation_seed": args.evaluation_seed,
                "teacher_evaluation": teacher,
                "teacher_gate": teacher_gate,
            },
        )
        print(f"checkpoint={checkpoint}")
        print(f"report={report_path}")
        print(json.dumps(bootstrap["grounder"]["gate"], indent=2))
        return
    ppo_state = {
        key: value.detach().cpu().clone()
        for key, value in trainer.policy.state_dict().items()
    }
    states = {"bootstrap": bootstrap_state}
    if args.total_timesteps > 0:
        states["puffer_ppo"] = ppo_state
    selected, screens, evaluation = evaluate_door_candidates(
        trainer,
        states,
        puffer_args,
        torch_pufferl,
        screen_steps=args.screen_steps,
        eval_steps=args.eval_steps,
        seed=args.evaluation_seed,
        agents=args.eval_agents,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = (
        args.output_dir
        / f"{args.env_name}_seed{args.seed}_{args.total_timesteps}.bin"
    )
    trainer.save_weights(checkpoint)
    model_size = trainer.model_size
    trainer.close()
    report = {
        "experiment": "D1-door",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "action_contract": CORRECTED_DOOR_ACTION_CONTRACT.to_report(),
        "policy_contract": door_policy_contract_report(
            hidden_size=settings.policy_hidden_size or 96,
            num_layers=settings.policy_num_layers or 1,
        ),
        "procedural_stream_contract": door_stream_contract_report(),
        "selected_stage": selected,
        "policy_parameters": model_size,
        "trainer": (
            "PufferLib 4 recurrent scheduled DAgger plus PuffeRL PPO"
            if args.total_timesteps > 0
            else "PufferLib 4 recurrent scheduled DAgger"
        ),
        "observation": (
            "64x48 gray4 current/delta/motion plus body velocity, gyro, "
            "gravity/body-z, altitude, takeoff-relative odometry/yaw, previous "
            "forward/yaw, and mission phase; no target token, pose, bearing, "
            "or map"
        ),
        "action": "forward speed and yaw rate; firmware stabilization below policy",
        "observability_checkpoint": str(args.observability_checkpoint),
        "observability_initialization": (
            "first eight validated 5x5 gray filters copied into the current-frame "
            "channel; equally weighted native and MuJoCo visibility supervision "
            "trains the grounder before action gradients train the compact encoder"
        ),
        "observability_frozen": True,
        "privileged_teacher_tail_excluded_by_encoder": True,
        "total_timesteps": args.total_timesteps,
        "training_obstacle_probability": args.obstacle_probability,
        "expanded_layouts": args.expanded_layouts,
        "hardware_camera_randomization": (
            args.hardware_camera_randomization
        ),
        "initial_checkpoint": (
            str(args.initial_checkpoint) if args.initial_checkpoint is not None else None
        ),
        "elapsed_s": elapsed,
        "bootstrap": bootstrap,
        "history": history,
        "evaluation_seed": args.evaluation_seed,
        "teacher_evaluation": teacher,
        "teacher_gate": teacher_gate,
        "candidate_screens": screens,
        "evaluation": evaluation,
        "simulation_gate": evaluation["gate"],
        "deployment_status": (
            "simulation gate passed; shadow-only validation next"
            if evaluation["gate"]["passed"]
            else "simulation gate failed; no live authority"
        ),
    }
    report_path = checkpoint.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"selected={selected}")
    print(f"checkpoint={checkpoint}")
    print(f"report={report_path}")
    print(json.dumps(evaluation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
