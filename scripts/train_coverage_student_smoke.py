from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.exploration.student_checkpoint import save_coverage_checkpoint
from flightrl.exploration.student_collection import (
    collect_matched_counterfactual_pair,
)
from flightrl.exploration.student_sequence import write_coverage_sequence
from flightrl.exploration.student_training import (
    CoverageTrainConfig,
    CoverageTrainingRejected,
    train_coverage_student,
)


TRAIN_SEED = 612
SELECTION_SEED = 613
SMOKE_CONFIG = CoverageTrainConfig(
    epochs=80,
    learning_rate=1.0e-2,
    tbptt_steps=1,
    seed=7,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the deterministic real-rendered coverage camera causal smoke. "
            "This does not evaluate closed-loop generalization or grant flight "
            "authority."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    targets = {
        "train_sequence": args.output_dir / "train_pair.npz",
        "selection_sequence": args.output_dir / "selection_pair.npz",
        "report": args.output_dir / "report.json",
        "checkpoint": args.output_dir / "student.pt",
    }
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        parser.error("refusing to overwrite existing artifacts: " + ", ".join(existing))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = collect_matched_counterfactual_pair(seed=TRAIN_SEED, split="train")
    selection = collect_matched_counterfactual_pair(
        seed=SELECTION_SEED,
        split="selection",
    )
    write_coverage_sequence(targets["train_sequence"], train)
    write_coverage_sequence(targets["selection_sequence"], selection)
    try:
        actor, report = train_coverage_student(train, selection, SMOKE_CONFIG)
    except CoverageTrainingRejected as error:
        report = error.report
        _write_report(targets["report"], report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2
    save_coverage_checkpoint(targets["checkpoint"], actor, report)
    _write_report(targets["report"], report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
