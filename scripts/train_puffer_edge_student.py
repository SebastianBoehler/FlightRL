from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from flightrl.artifact_paths import require_distinct_artifact_paths
from flightrl.evidence_scope import file_identity
from flightrl.puffer4_edge_checkpoint import (
    build_edge_checkpoint_payload,
    save_edge_checkpoint,
)
from flightrl.puffer4_edge_sequence import load_edge_sequence_dataset
from flightrl.puffer4_edge_training import (
    EdgeTrainConfig,
    EdgeTrainingRejected,
    train_edge_student,
)
from flightrl.puffer4_edge_training_sources import edge_training_source_identity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the exact recurrent edge-v3 door student"
    )
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--selection-dataset", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/edge_v3/edge_door_student.pt"),
    )
    parser.add_argument(
        "--training-report",
        type=Path,
        default=Path("artifacts/edge_v3/edge_door_training.json"),
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--warmup-batch-size", type=int, default=512)
    parser.add_argument("--perception-learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--tbptt-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    paths = require_distinct_artifact_paths(
        train_dataset=args.train_dataset,
        selection_dataset=args.selection_dataset,
        checkpoint=args.checkpoint,
        training_report=args.training_report,
    )
    args.train_dataset = paths["train_dataset"]
    args.selection_dataset = paths["selection_dataset"]
    args.checkpoint = paths["checkpoint"]
    args.training_report = paths["training_report"]
    source_identity = edge_training_source_identity()
    dataset_identities = _training_dataset_identities(
        args.train_dataset,
        args.selection_dataset,
    )

    config = EdgeTrainConfig(
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        warmup_batch_size=args.warmup_batch_size,
        perception_learning_rate=args.perception_learning_rate,
        learning_rate=args.learning_rate,
        tbptt_steps=args.tbptt_steps,
        seed=args.seed,
    )
    train = load_edge_sequence_dataset(
        args.train_dataset,
        verify_execution_trace=False,
    )
    selection = load_edge_sequence_dataset(
        args.selection_dataset,
        verify_execution_trace=False,
    )
    _require_unchanged_training_inputs(
        source_identity,
        dataset_identities,
        "after dataset load",
    )
    rejected = False
    try:
        actor, report = train_edge_student(train, selection, config)
    except EdgeTrainingRejected as exc:
        report = exc.report
        rejected = True
    _require_unchanged_training_inputs(
        source_identity,
        dataset_identities,
        "before report",
    )
    report["datasets"] = dataset_identities
    report["native_build_fingerprint"] = train.metadata[
        "native_build_fingerprint"
    ]
    report["source_identity"] = source_identity
    args.training_report.parent.mkdir(parents=True, exist_ok=True)
    args.training_report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    if rejected:
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        print(
            f"training_rejected_report={args.training_report.resolve()}",
            file=sys.stderr,
        )
        return 2
    checkpoint = build_edge_checkpoint_payload(
        actor,
        trained_target_ids=[0],
        dataset=args.selection_dataset,
        training_report=args.training_report,
    )
    _require_unchanged_training_inputs(
        source_identity,
        dataset_identities,
        "before checkpoint save",
    )
    save_edge_checkpoint(checkpoint, args.checkpoint)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"checkpoint={args.checkpoint.resolve()}")
    print(f"training_report={args.training_report.resolve()}")
    return 0


def _require_unchanged_training_sources(
    captured: dict[str, dict[str, str]],
    stage: str,
) -> None:
    if edge_training_source_identity() != captured:
        raise RuntimeError(f"edge training sources changed {stage}")


def _training_dataset_identities(
    train: Path,
    selection: Path,
) -> dict[str, dict[str, str]]:
    return {
        "train": file_identity(train),
        "selection": file_identity(selection),
    }


def _require_unchanged_training_inputs(
    source_identity: dict[str, dict[str, str]],
    dataset_identities: dict[str, dict[str, str]],
    stage: str,
) -> None:
    _require_unchanged_training_sources(source_identity, stage)
    current = _training_dataset_identities(
        Path(dataset_identities["train"]["path"]),
        Path(dataset_identities["selection"]["path"]),
    )
    if current != dataset_identities:
        raise RuntimeError(f"edge training dataset changed {stage}")


if __name__ == "__main__":
    raise SystemExit(main())
