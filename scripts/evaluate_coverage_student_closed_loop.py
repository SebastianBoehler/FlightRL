from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.exploration.student_checkpoint import load_coverage_checkpoint
from flightrl.exploration.student_closed_loop import (
    evaluate_coverage_student_closed_loop,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a coverage student under clean, frozen, and fixed-permuted "
            "camera histories in held-out MuJoCo rooms."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=620)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.seed_start < 0 or args.episodes < 2 or args.steps <= 0:
        parser.error(
            "seed start must be non-negative, steps positive, and episodes at least two"
        )
    actor, training_report = load_coverage_checkpoint(args.checkpoint)
    scene_ids = tuple(range(args.seed_start, args.seed_start + args.episodes))
    training_scenes = {
        int(seed)
        for split in training_report["datasets"].values()
        for seed in split["scene_ids"]
    }
    if training_scenes & set(scene_ids):
        parser.error("closed-loop scene IDs overlap the checkpoint datasets")
    report = evaluate_coverage_student_closed_loop(
        actor,
        scene_ids=scene_ids,
        maximum_steps=args.steps,
    )
    report["checkpoint_state_sha256"] = training_report[
        "selected_actor_state_sha256"
    ]
    report["training_closed_loop_evaluated"] = training_report[
        "closed_loop_evaluated"
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if report["closed_loop_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
