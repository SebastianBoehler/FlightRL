from __future__ import annotations

import argparse
import hashlib
import json
from math import isfinite
from pathlib import Path
import subprocess
import sys

from flightrl.puffer4_config import Puffer4ExportSettings
from flightrl.puffer4_door_contract import (
    PRIVILEGED_DOOR_TEACHER_ACTION_CONTRACT,
)
from flightrl.puffer4_door_export import export_fixed_door_assets
from flightrl.puffer4_door_mission import FIXED_DOOR_MISSION_METRIC_V1
from flightrl.puffer4_door_runner import (
    build_environment,
    load_puffer,
    native_source_paths,
    verify_native_build,
)
from flightrl.puffer4_door_teacher import (
    evaluate_privileged_door_teacher,
    privileged_teacher_gate,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUFFER = ROOT.parent / "PufferLib-4-flightrl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the privileged fixed-door simulation teacher"
    )
    parser.add_argument("--puffer-root", type=Path, default=DEFAULT_PUFFER)
    parser.add_argument(
        "--env-name",
        default="flightrl_fixed_door_privileged_teacher",
    )
    parser.add_argument("--agents", type=int, default=128)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=10_011)
    parser.add_argument(
        "--obstacle-probability",
        type=float,
        default=0.0,
        help="Episode obstacle probability in [0, 1]",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.agents <= 0 or args.steps <= 0 or args.seed < 0:
        parser.error("agents/steps must be positive and seed must be nonnegative")
    if (
        not isfinite(args.obstacle_probability)
        or not 0.0 <= args.obstacle_probability <= 1.0
    ):
        parser.error("obstacle probability must be finite and in [0, 1]")

    settings = Puffer4ExportSettings(
        env_name=args.env_name,
        total_agents=args.agents,
        num_buffers=1,
        num_threads=8,
        train_seed=11,
    )
    export_fixed_door_assets(args.puffer_root, settings)
    fingerprint = (
        verify_native_build(args.puffer_root, args.env_name)
        if args.skip_build
        else build_environment(args.puffer_root, args.env_name)
    )
    puffer_args, torch_pufferl = load_puffer(args.puffer_root, args.env_name)
    overrides = {
        "obstacle_probability": args.obstacle_probability,
        "layout_diversity": 1.0,
    }
    puffer_args["env"].update(overrides)
    FIXED_DOOR_MISSION_METRIC_V1.verify_env(puffer_args["env"])
    metrics = evaluate_privileged_door_teacher(
        puffer_args,
        torch_pufferl,
        steps=args.steps,
        seed=args.seed,
        agents=args.agents,
    )
    report = {
        "schema": "flightrl.fixed_door.privileged_teacher.v2",
        "status": "complete",
        "authority": {
            "scope": "desktop_privileged_teacher",
            "learned_policy": False,
            "checkpoint": False,
            "deployment_authority": False,
        },
        "edge_v3_alignment": {
            "action_envelope": True,
            "observation_contract": False,
            "runtime_cadence_measured_and_bound": False,
            "deployable_policy": False,
        },
        "environment": {
            "name": args.env_name,
            "agents": args.agents,
            "steps": args.steps,
            "seed": args.seed,
            "overrides": overrides,
            "control_dt_s": puffer_args["env"]["control_dt"],
            "max_episode_steps": puffer_args["env"]["max_episode_steps"],
            "max_episode_duration_s": (
                puffer_args["env"]["control_dt"]
                * puffer_args["env"]["max_episode_steps"]
            ),
        },
        "mission_metric": FIXED_DOOR_MISSION_METRIC_V1.to_report(),
        "teacher_action_contract": (
            PRIVILEGED_DOOR_TEACHER_ACTION_CONTRACT.to_report()
        ),
        "metrics": metrics,
        "gate": privileged_teacher_gate(metrics),
        "provenance": provenance(args, fingerprint),
    }
    output = args.output or Path(
        f"artifacts/evidence/fixed_door_privileged_teacher_seed{args.seed}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={output.resolve()}")


def provenance(args: argparse.Namespace, native_fingerprint: dict) -> dict:
    sources = {
        Path(__file__).resolve(),
        ROOT / "src/flightrl/puffer4_door_teacher.py",
        ROOT / "src/flightrl/puffer4_door_mission.py",
        ROOT / "src/flightrl/puffer4_door_contract.py",
        ROOT / "src/flightrl/puffer4_door_sections.py",
        ROOT / "src/flightrl/puffer4_door_export.py",
        ROOT / "src/flightrl/puffer4_door_runner.py",
        *native_source_paths(args.puffer_root, args.env_name),
    }
    return {
        "command": list(sys.argv),
        "flightrl": repository_identity(ROOT),
        "puffer": repository_identity(args.puffer_root.resolve()),
        "source_sha256": {
            str(path.resolve()): sha256(path)
            for path in sorted(sources, key=lambda item: str(item.resolve()))
        },
        "native_build_fingerprint": native_fingerprint,
    }


def repository_identity(root: Path) -> dict[str, str | None]:
    return {
        "path": str(root.resolve()),
        "head": git_output(root, "rev-parse", "HEAD"),
        "tracked_diff_sha256": git_diff_sha256(root),
    }


def git_output(root: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_diff_sha256(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=root,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest() if result.returncode == 0 else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
