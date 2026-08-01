from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import sysconfig
from time import perf_counter

import torch

from flightrl.puffer4_config import Puffer4ExportSettings
from flightrl.puffer4_door_contract import CORRECTED_DOOR_ACTION_CONTRACT
from flightrl.puffer4_door_asymmetric import (
    DoorAsymmetricConfig,
    DoorAsymmetricTrainer,
)
from flightrl.puffer4_door_checkpoint import (
    initialize_door_policy,
)
from flightrl.puffer4_door_evaluation import evaluate_door_candidates
from flightrl.puffer4_door_export import (
    DOOR_NATIVE_FILES,
    export_fixed_door_assets,
)
from flightrl.puffer4_door_imitation import (
    bootstrap_door_policy,
    freeze_door_grounder,
)
from flightrl.puffer4_door_policy_contract import door_policy_contract_report
from flightrl.puffer4_door_runner import (
    build_environment,
    load_puffer,
    native_build_marker_path,
    verify_native_build,
)
from flightrl.puffer4_door_provenance import (
    build_door_run_provenance,
    build_file_manifest,
)
from flightrl.puffer4_door_stream_contract import door_stream_contract_report
from flightrl.puffer4_door_training import (
    evaluate_door_teacher,
    fixed_door_teacher_gate,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUFFER = ROOT.parent / "PufferLib-4-flightrl"
DEFAULT_SOURCE = (
    ROOT
    / "artifacts/puffer_fixed_door_d1_v53_asymmetric_conservative262k"
    / "flightrl_fixed_door_d1_seed11_262144.bin"
)


def main() -> None:
    run_started = perf_counter()
    started_at_utc = datetime.now(timezone.utc).isoformat()
    command = [sys.executable, *sys.argv]
    flightrl_source_sha256 = build_file_manifest(
        ROOT,
        [
            Path(__file__),
            *ROOT.glob("src/flightrl/puffer4_door*.py"),
            *ROOT.glob("src/flightrl/native/native_door*"),
        ],
    )
    parser = argparse.ArgumentParser(
        description="Train the recurrent door actor with DAgger-regularized asymmetric PPO"
    )
    parser.add_argument("--puffer-root", type=Path, default=DEFAULT_PUFFER)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--fresh-control", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/puffer_fixed_door_d1_asymmetric",
    )
    parser.add_argument("--env-name", default="flightrl_fixed_door_d1")
    parser.add_argument("--rollouts", type=int, default=64)
    parser.add_argument("--bootstrap-updates", type=int, default=64)
    parser.add_argument("--bootstrap-learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--bootstrap-max-policy-rollin", type=float, default=0.0)
    parser.add_argument("--agents", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--imitation-coefficient", type=float, default=0.25)
    parser.add_argument("--policy-logstd", type=float, default=-3.0)
    parser.add_argument("--optimization-epochs", type=int, default=2)
    parser.add_argument("--minibatch-agents", type=int, default=32)
    parser.add_argument("--screen-steps", type=int, default=1400)
    parser.add_argument("--eval-steps", type=int, default=3000)
    parser.add_argument("--evaluation-seed", type=int, default=10_011)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    source_report = args.source_report or args.source_checkpoint.with_suffix(
        ".report.json"
    )
    verify_source(args.source_checkpoint, source_report)

    settings = Puffer4ExportSettings(
        env_name=args.env_name,
        total_agents=args.agents,
        num_buffers=1,
        num_threads=8,
        policy_hidden_size=96,
        policy_num_layers=1,
        train_seed=args.seed,
    )
    export_result = export_fixed_door_assets(args.puffer_root, settings)
    if not args.skip_build:
        build_environment(args.puffer_root, args.env_name)
    puffer_args, torch_pufferl = load_puffer(args.puffer_root, args.env_name)
    native_build_fingerprint = verify_native_build(args.puffer_root, args.env_name)
    puffer_args["env"]["obstacle_probability"] = 0.0
    puffer_args["env"]["layout_diversity"] = 1.0
    CORRECTED_DOOR_ACTION_CONTRACT.verify_env(puffer_args["env"])
    puffer_args["vec"]["total_agents"] = args.agents
    teacher = evaluate_door_teacher(
        puffer_args,
        torch_pufferl,
        steps=args.eval_steps,
        seed=args.evaluation_seed,
        agents=args.agents,
    )
    teacher_gate = fixed_door_teacher_gate(teacher)
    if not teacher_gate["passed"]:
        raise RuntimeError(
            f"observation-matched teacher gate failed: {teacher_gate['failures']}"
    )
    vec = torch_pufferl._C.create_vec(puffer_args, torch_pufferl._C.gpu)
    source_state = torch.load(
        args.source_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    policy, migration = initialize_door_policy(
        lambda: torch_pufferl.load_policy(puffer_args, vec),
        source_state,
        seed=args.seed,
        fresh_control=args.fresh_control,
    )
    freeze_door_grounder(policy)
    initial_state = deepcopy(policy.state_dict())
    bootstrap = bootstrap_door_policy(
        policy,
        vec,
        torch_pufferl,
        updates=args.bootstrap_updates,
        learning_rate=args.bootstrap_learning_rate,
        max_policy_rollin=args.bootstrap_max_policy_rollin,
    )
    bootstrap_state = deepcopy(policy.state_dict())
    config = DoorAsymmetricConfig(
        horizon=args.horizon,
        learning_rate=args.learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        imitation_coefficient=args.imitation_coefficient,
        policy_logstd=args.policy_logstd,
        optimization_epochs=args.optimization_epochs,
        minibatch_agents=args.minibatch_agents,
    )
    trainer = DoorAsymmetricTrainer(policy, vec, torch_pufferl, config)
    history = trainer.train(args.rollouts)
    trained_state = deepcopy(policy.state_dict())
    candidate_states = {
        "source": initial_state,
        "bootstrap": bootstrap_state,
        "asymmetric_ppo": trained_state,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_checkpoints = {}
    for name, state in candidate_states.items():
        path = args.output_dir / f"candidate-{name}.bin"
        torch.save(state, path)
        candidate_checkpoints[name] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }
    selected, screens, evaluation = evaluate_door_candidates(
        trainer,
        candidate_states,
        puffer_args,
        torch_pufferl,
        screen_steps=args.screen_steps,
        eval_steps=args.eval_steps,
        seed=args.evaluation_seed,
        agents=args.agents,
    )
    training_steps = (
        args.bootstrap_updates + args.rollouts
    ) * args.agents * args.horizon
    checkpoint = args.output_dir / (
        f"{args.env_name}_seed{args.seed}_{training_steps}.bin"
    )
    torch.save(policy.state_dict(), checkpoint)
    generated_files = [
        export_result.config_path,
        export_result.env_dir / "binding.c",
        *(export_result.env_dir / name for name in DOOR_NATIVE_FILES),
        args.puffer_root
        / "pufferlib"
        / f"_C{sysconfig.get_config_var('EXT_SUFFIX')}",
        args.puffer_root / "pufferlib" / "torch_pufferl.py",
        native_build_marker_path(args.puffer_root),
    ]
    run_provenance = build_door_run_provenance(
        command=command,
        started_at_utc=started_at_utc,
        elapsed_wall_s=perf_counter() - run_started,
        source_report=source_report,
        flightrl_root=ROOT,
        flightrl_source_sha256=flightrl_source_sha256,
        puffer_root=args.puffer_root,
        generated_files=generated_files,
        native_build_fingerprint=native_build_fingerprint,
    )
    report = {
        "experiment": "D1-door-asymmetric-ppo",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "source_checkpoint": str(args.source_checkpoint),
        "source_checkpoint_sha256": file_sha256(args.source_checkpoint),
        "run_provenance": run_provenance,
        "source_migration": migration,
        "action_contract": CORRECTED_DOOR_ACTION_CONTRACT.to_report(),
        "policy_contract": door_policy_contract_report(
            hidden_size=settings.policy_hidden_size or 96,
            num_layers=settings.policy_num_layers or 1,
        ),
        "procedural_stream_contract": door_stream_contract_report(),
        "bootstrap": bootstrap,
        "teacher_evaluation": teacher,
        "teacher_gate": teacher_gate,
        "candidate_checkpoints": candidate_checkpoints,
        "selected_stage": selected,
        "trainer": (
            "persistent MinGRU actor plus privileged critic, "
            "on-policy teacher regularization, and PPO"
        ),
        "controls_drone": False,
        "policy_parameters": sum(
            parameter.numel() for parameter in policy.parameters()
        ),
        "critic_parameters_training_only": sum(
            parameter.numel() for parameter in trainer.critic.parameters()
        ),
        "config": vars(args) | {"trainer": asdict(config)},
        "history": history,
        "candidate_screens": screens,
        "evaluation": evaluation,
        "simulation_gate": evaluation["gate"],
        "deployment_status": (
            "simulation gate passed; hardware shadow next"
            if evaluation["gate"]["passed"]
            else "simulation gate failed; no live authority"
        ),
    }
    report["config"] = json_safe(report["config"])
    report_path = checkpoint.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    vec.close()
    print(f"selected={selected}")
    print(f"checkpoint={checkpoint}")
    print(f"report={report_path}")
    print(json.dumps(evaluation, indent=2, sort_keys=True))


def verify_source(checkpoint: Path, report_path: Path) -> None:
    report = json.loads(report_path.read_text())
    if report.get("checkpoint_sha256") != file_sha256(checkpoint):
        raise ValueError("source checkpoint does not match its training report")
    if report.get("selected_stage") not in {"bootstrap", "asymmetric_ppo"}:
        raise ValueError("source checkpoint is not a selected door policy")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    return value


if __name__ == "__main__":
    main()
