from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.semantic.shadow_gate import (
    replay_shadow_run,
    semantic_shadow_gate,
    semantic_translation_shadow_gate,
    write_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate a semantic checkpoint against a recorded AI Deck flight"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir or args.checkpoint.parent / (
        args.checkpoint.stem + "_live_gate"
    )
    rows = replay_shadow_run(
        args.run_dir,
        args.checkpoint,
        args.training_report,
    )
    suppressed = replay_shadow_run(
        args.run_dir,
        args.checkpoint,
        args.training_report,
        suppress_detections=True,
    )
    metrics = semantic_shadow_gate(rows, suppressed)
    translation_metrics = semantic_translation_shadow_gate(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "replay.csv", rows)
    write_rows(output_dir / "detection_suppressed.csv", suppressed)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "training_report": str(args.training_report.resolve()),
        "run_dir": str(args.run_dir.resolve()),
        **metrics,
        **translation_metrics,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
