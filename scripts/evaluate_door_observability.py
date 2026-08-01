from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import sys
import time

import numpy as np
import torch

from flightrl.mujoco import is_mujoco_available
from flightrl.mujoco.door_observability import collect_synthetic_door_dataset
from flightrl.semantic.door_observability import (
    DoorObservabilityGate,
    decode_observability,
    observability_metrics,
)
from flightrl.semantic.door_observability_training import (
    DoorObservabilityTrainingConfig,
    train_door_observability,
)
from flightrl.semantic.door_real_evidence import load_real_door_evidence
from flightrl.semantic.frame_integrity import load_frame_integrity_registry


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    args = parse_args()
    if not is_mujoco_available():
        raise SystemExit("MuJoCo is required for the door observability gate")
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_seeds = tuple(range(args.seed, args.seed + args.train_rooms))
    evaluation_seeds = tuple(
        range(
            args.seed + 10_000,
            args.seed + 10_000 + args.evaluation_rooms,
        )
    )
    started = time.perf_counter()
    train = collect_synthetic_door_dataset(
        room_seeds=train_seeds,
        samples_per_room=args.samples_per_room,
        seed=args.seed + 20_000,
    )
    evaluation = collect_synthetic_door_dataset(
        room_seeds=evaluation_seeds,
        samples_per_room=args.samples_per_room,
        seed=args.seed + 30_000,
    )
    collection_seconds = time.perf_counter() - started
    config = DoorObservabilityTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    device = resolve_device(args.device)
    training_started = time.perf_counter()
    result = train_door_observability(
        train_frames=train.frames,
        train_labels=train.labels,
        validation_frames=evaluation.frames,
        validation_labels=evaluation.labels,
        config=config,
        device=device,
    )
    training_seconds = time.perf_counter() - training_started
    synthetic_metrics = observability_metrics(
        result.validation_predictions,
        evaluation.labels,
    )
    registry_path = args.integrity_registry.resolve()
    registry = load_frame_integrity_registry(registry_path, root=ROOT)
    real_positive = None
    real_negative = None
    real_summary: dict[str, object] = {
        "status": "missing_labeled_door_positive_and_negative_sequences",
    }
    if args.real_manifest is not None:
        real = load_real_door_evidence(
            args.real_manifest,
            root=ROOT,
            integrity_registry=registry,
        )
        result.model.eval()
        with torch.no_grad():
            raw = result.model(torch.from_numpy(real.frames).to(device))
            real_predictions = decode_observability(raw).cpu().numpy()
        positive = real.labels[:, 0] > 0.5
        negative = ~positive
        if np.any(positive):
            real_positive = observability_metrics(
                real_predictions[positive],
                real.labels[positive],
            )
        if np.any(negative):
            real_negative = observability_metrics(
                real_predictions[negative],
                real.labels[negative],
            )
        real_summary = {
            "status": "evaluated",
            "manifest": str(args.real_manifest.resolve()),
            "sample_count": int(real.labels.shape[0]),
            "positive_metrics": (
                None if real_positive is None else real_positive.to_dict()
            ),
            "negative_metrics": (
                None if real_negative is None else real_negative.to_dict()
            ),
        }
    gate = DoorObservabilityGate()
    gate_result = gate.evaluate(
        synthetic=synthetic_metrics,
        real_positive=real_positive,
        real_negative=real_negative,
    )
    checkpoint_path = output_dir / "door_observability.pt"
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu()
                for key, value in result.model.state_dict().items()
            },
            "frame_contract": {
                "width": 64,
                "height": 48,
                "channels": 1,
                "quantization_levels": 16,
            },
            "training_config": asdict(config),
            "train_room_seeds": train_seeds,
            "evaluation_room_seeds": evaluation_seeds,
        },
        checkpoint_path,
    )
    report = {
        "schema_version": 1,
        "experiment": "fixed_door_observability_pretest",
        "status": gate_result.status,
        "frame_contract": {
            "width": 64,
            "height": 48,
            "channels": 1,
            "bit_depth": 4,
            "input": "raw grayscale camera frame",
        },
        "target_contract": {
            "category": "door",
            "policy_target_input": False,
            "labels": "MuJoCo segmentation used for supervision only",
        },
        "training": {
            "device": device,
            "config": asdict(config),
            "train_room_seeds": train_seeds,
            "evaluation_room_seeds": evaluation_seeds,
            "samples_per_room": args.samples_per_room,
            "train_samples": int(train.frames.shape[0]),
            "evaluation_samples": int(evaluation.frames.shape[0]),
            "train_positive_samples": int(np.sum(train.labels[:, 0])),
            "evaluation_positive_samples": int(np.sum(evaluation.labels[:, 0])),
            "final_train_loss": result.final_train_loss,
            "collection_seconds": collection_seconds,
            "training_seconds": training_seconds,
        },
        "synthetic_metrics": synthetic_metrics.to_dict(),
        "thresholds": asdict(gate),
        "gate": gate_result.to_dict(),
        "real_evidence": {
            **real_summary,
            "policy": "only explicitly frame-safe datasets may be labeled",
            "integrity_registry": str(registry_path),
            "registered_datasets": [
                {
                    "path": str(record.path.relative_to(ROOT)),
                    "status": record.status,
                    "evidence": record.evidence,
                }
                for record in registry.records
            ],
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256(checkpoint_path),
            "parameter_count": sum(
                parameter.numel() for parameter in result.model.parameters()
            ),
        },
        "next_gate": (
            "label frame-safe real door-positive and door-negative sequences"
            if gate_result.synthetic_pass
            else "improve simulator appearance or camera resolution before RL"
        ),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output_dir / "report.md").write_text(render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the fixed-door camera observability pretest."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/semantic/door-observability-64x48-20260729",
    )
    parser.add_argument(
        "--integrity-registry",
        type=Path,
        default=ROOT / "configs/semantic/aideck_frame_integrity.json",
    )
    parser.add_argument(
        "--real-manifest",
        type=Path,
        help="reviewed real-frame labels, with paths relative to the repository root",
    )
    parser.add_argument("--train-rooms", type=int, default=24)
    parser.add_argument("--evaluation-rooms", type=int, default=8)
    parser.add_argument("--samples-per-room", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    for name in ("train_rooms", "evaluation_rooms", "samples_per_room", "epochs"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_markdown(report: dict[str, object]) -> str:
    metrics = report["synthetic_metrics"]
    gate = report["gate"]
    training = report["training"]
    checkpoint = report["checkpoint"]
    return (
        "# Fixed-door observability pretest\n\n"
        f"- Status: `{report['status']}`\n"
        f"- Synthetic AUROC: `{metrics['visibility_auroc']:.4f}`\n"
        "- Synthetic median centroid error: "
        f"`{metrics['centroid_median_error_widths']:.4f}` image widths\n"
        f"- Train/evaluation samples: `{training['train_samples']}` / "
        f"`{training['evaluation_samples']}`\n"
        f"- Collection/training: `{training['collection_seconds']:.2f}s` / "
        f"`{training['training_seconds']:.2f}s`\n"
        f"- Parameters: `{checkpoint['parameter_count']}`\n"
        f"- Checkpoint SHA-256: `{checkpoint['sha256']}`\n"
        f"- Gate failures: `{gate['failures']}`\n"
        f"- Next gate: {report['next_gate']}\n"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
