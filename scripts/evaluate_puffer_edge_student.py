from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.artifact_paths import require_distinct_artifact_paths
from flightrl.evidence_scope import (
    file_identity,
    require_existing_file_identity,
)
from flightrl.puffer4_config import Puffer4ExportSettings
from flightrl.puffer4_door_runner import load_puffer, verify_native_build
from flightrl.puffer4_edge_checkpoint import load_edge_checkpoint
from flightrl.puffer4_edge_evaluation import evaluate_edge_student
from flightrl.puffer4_edge_evaluation_gate import (
    EDGE_EVALUATION_PROFILES,
    EDGE_EVALUATION_SCHEMA,
)
from flightrl.puffer4_edge_evaluation_metrics import (
    require_evaluation_metric_consistency,
)
from flightrl.puffer4_edge_native_build import (
    require_current_edge_native_build_fingerprint,
    require_matching_edge_native_build_fingerprints,
)
from flightrl.puffer4_edge_sequence import load_edge_sequence_dataset
from flightrl.puffer4_edge_student_export import (
    EDGE_STUDENT_NATIVE_FILES,
    write_edge_student_config,
)
from flightrl.puffer4_edge_training import EDGE_TRAINING_REPORT_SCHEMA


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUFFER = ROOT.parent / "PufferLib-4-edge-v3"
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run disjoint closed-loop evaluation of an edge-v3 door student"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--puffer-root", type=Path, default=DEFAULT_PUFFER)
    parser.add_argument("--env-name", default="flightrl_edge_v3_door_student")
    parser.add_argument("--agents", type=int, default=128)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--fail-on-gate", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/edge_v3/edge_door_held_out.json"),
    )
    args = parser.parse_args(argv)
    if args.agents <= 0 or args.steps <= 0:
        parser.error("agents and steps must be positive")
    config_path = (
        args.puffer_root.expanduser().resolve()
        / "config"
        / f"{args.env_name}.ini"
    )
    paths = require_distinct_artifact_paths(
        checkpoint=args.checkpoint,
        output=args.output,
        environment_config=config_path,
    )
    args.checkpoint = paths["checkpoint"]
    args.output = paths["output"]
    config_path = paths["environment_config"]
    source_identity = _evaluation_source_identity()

    checkpoint_identity = file_identity(args.checkpoint)
    actor, checkpoint = load_edge_checkpoint(args.checkpoint)
    if file_identity(args.checkpoint) != checkpoint_identity:
        raise RuntimeError("edge evaluation checkpoint changed while it was loaded")
    if checkpoint.trained_target_ids != (0,):
        raise SystemExit("first edge door evaluator requires a door-only checkpoint")
    training = _load_training_report(checkpoint.training_identity)
    require_distinct_artifact_paths(
        checkpoint=args.checkpoint,
        output=args.output,
        environment_config=config_path,
        training_report=checkpoint.training_identity["path"],
        **_training_dataset_paths(training),
    )
    _require_final_seed_disjointness(training)
    fingerprint = _verify_build(args.puffer_root, args.env_name)
    require_matching_edge_native_build_fingerprints(
        checkpoint.native_build_fingerprint,
        fingerprint,
    )
    require_current_edge_native_build_fingerprint(
        fingerprint, expected=checkpoint.native_build_fingerprint
    )
    settings = Puffer4ExportSettings(
        env_name=args.env_name,
        total_agents=args.agents,
        num_buffers=1,
        num_threads=8,
        policy_hidden_size=checkpoint.hidden_size,
        train_seed=17,
    )
    written_config_path = write_edge_student_config(args.puffer_root, settings).resolve()
    if written_config_path != config_path:
        raise RuntimeError("edge evaluator wrote an unexpected environment config path")
    environment_config_identity = file_identity(config_path)
    puffer_args, torch_pufferl = load_puffer(
        args.puffer_root,
        args.env_name,
        EDGE_STUDENT_NATIVE_FILES,
    )
    require_current_edge_native_build_fingerprint(
        fingerprint, expected=checkpoint.native_build_fingerprint
    )
    if file_identity(config_path) != environment_config_identity:
        raise RuntimeError("edge evaluation environment config changed while loading")
    if _evaluation_source_identity() != source_identity:
        raise RuntimeError("edge evaluation sources changed before evaluation")
    records = {}
    for name, seed, appearance_seed, profile in EDGE_EVALUATION_PROFILES:
        record = evaluate_edge_student(
            puffer_args,
            torch_pufferl,
            actor,
            steps=args.steps,
            agents=args.agents,
            seed=seed,
            appearance_seed=appearance_seed,
            profile=profile,
        )
        require_evaluation_metric_consistency(
            record["metrics"],
            configuration=profile,
            steps=args.steps,
            agents=args.agents,
        )
        records[name] = record
    failures = [name for name, value in records.items() if not value["gate"]["passed"]]
    report = {
        "schema": EDGE_EVALUATION_SCHEMA,
        "status": "complete",
        "scope": "desktop_simulation_held_out",
        "checkpoint_identity": checkpoint_identity,
        "policy_contract_sha256": checkpoint.policy_contract_sha256,
        "evaluated_target_ids": [0],
        "profiles": records,
        "gate": {"passed": not failures, "failures": failures},
        "native_build_fingerprint": fingerprint,
        "environment_config_identity": environment_config_identity,
        "source_identity": source_identity,
        "authority": "none",
        "deployment_authority": False,
        "hardware_approved": False,
        "controls_drone": False,
    }
    _require_bound_file_unchanged(
        checkpoint_identity, "checkpoint", "during evaluation"
    )
    _require_bound_file_unchanged(
        environment_config_identity, "environment config", "during evaluation"
    )
    if _evaluation_source_identity() != source_identity:
        raise RuntimeError("edge evaluation sources changed during evaluation")
    require_current_edge_native_build_fingerprint(
        fingerprint, expected=checkpoint.native_build_fingerprint
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={args.output.resolve()}")
    if args.fail_on_gate and failures:
        raise SystemExit(2)


def _load_training_report(identity: dict[str, str]) -> dict:
    require_existing_file_identity(identity, label="edge training report")
    payload = json.loads(Path(identity["path"]).read_text())
    if payload.get("schema") != EDGE_TRAINING_REPORT_SCHEMA:
        raise ValueError("edge checkpoint training report schema is incompatible")
    return payload


def _evaluation_source_identity() -> dict[str, dict[str, str]]:
    return {
        "script": file_identity(Path(__file__)),
        "artifact_paths": file_identity(ROOT / "src/flightrl/artifact_paths.py"),
        "evaluator": file_identity(ROOT / "src/flightrl/puffer4_edge_evaluation.py"),
        "gate": file_identity(ROOT / "src/flightrl/puffer4_edge_evaluation_gate.py"),
        "counts": file_identity(
            ROOT / "src/flightrl/puffer4_edge_evaluation_counts.py"
        ),
        "metrics": file_identity(
            ROOT / "src/flightrl/puffer4_edge_evaluation_metrics.py"
        ),
        "exporter": file_identity(
            ROOT / "src/flightrl/puffer4_edge_student_export.py"
        ),
        "sections": file_identity(
            ROOT / "src/flightrl/puffer4_edge_student_sections.py"
        ),
        "door_sections": file_identity(ROOT / "src/flightrl/puffer4_door_sections.py"),
        "config": file_identity(ROOT / "src/flightrl/puffer4_config.py"),
        "native_identity": file_identity(
            ROOT / "src/flightrl/puffer4_edge_native_build.py"
        ),
    }


def _require_final_seed_disjointness(training: dict) -> None:
    identities = training.get("datasets")
    if not isinstance(identities, dict) or set(identities) != {"train", "selection"}:
        raise ValueError("edge training report dataset identities are missing")
    used_physical = set()
    used_appearance = set()
    for split in ("train", "selection"):
        identity = require_existing_file_identity(
            identities[split],
            label=f"edge {split} dataset",
        )
        dataset = load_edge_sequence_dataset(
            identity["path"],
            verify_execution_trace=False,
        )
        used_physical.add(dataset.metadata["base_seed"])
        used_appearance.add(dataset.metadata["appearance_seed"])
    final_physical = {profile[1] for profile in EDGE_EVALUATION_PROFILES}
    final_appearance = {profile[2] for profile in EDGE_EVALUATION_PROFILES}
    if (
        used_physical & final_physical
        or used_appearance & final_appearance
        or len(final_physical) != len(EDGE_EVALUATION_PROFILES)
        or len(final_appearance) != len(EDGE_EVALUATION_PROFILES)
    ):
        raise ValueError("edge final evaluation seeds overlap training or selection")


def _training_dataset_paths(training: object) -> dict[str, str]:
    if not isinstance(training, dict):
        return {}
    identities = training.get("datasets")
    if not isinstance(identities, dict):
        return {}
    result = {}
    for split in ("train", "selection"):
        identity = identities.get(split)
        if isinstance(identity, dict) and isinstance(identity.get("path"), str):
            result[f"{split}_dataset"] = identity["path"]
    return result


def _verify_build(root: Path, env_name: str) -> dict:
    return verify_native_build(root, env_name, EDGE_STUDENT_NATIVE_FILES)


def _require_bound_file_unchanged(
    identity: dict[str, str],
    label: str,
    stage: str,
) -> None:
    if file_identity(identity["path"]) != identity:
        raise RuntimeError(f"edge evaluation {label} changed {stage}")


if __name__ == "__main__":
    main()
