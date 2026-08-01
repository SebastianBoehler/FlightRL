from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flightrl.semantic.shadow_gate import replay_shadow_run, write_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a semantic AI Deck run through a non-actuating Puffer policy"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--assumed-target-distance-m", type=float, default=2.0)
    args = parser.parse_args()

    rows = replay_shadow_run(
        args.run_dir,
        args.checkpoint,
        args.training_report,
        assumed_target_distance_m=args.assumed_target_distance_m,
    )

    output = args.output or args.run_dir / "puffer_shadow.csv"
    write_rows(output, rows)
    action_l2 = [
        float(
            np.linalg.norm(
                [
                    row["action_vx"],
                    row["action_vy"],
                    row["action_vz"],
                    row["action_yaw"],
                ]
            )
        )
        for row in rows
    ]
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "training_report": str(args.training_report.resolve()),
        "controls_drone": False,
        "monitor_only": True,
        "processed_frames": len(rows),
        "frames_with_detection": sum(bool(row["target_detected"]) for row in rows),
        "mean_action_l2": float(np.mean(action_l2)) if action_l2 else None,
        "max_action_l2": max(action_l2, default=None),
        "assumed_target_distance_m": args.assumed_target_distance_m,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"shadow_rows={len(rows)}")
    print("controls_drone=False")
    print(f"output={output}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
