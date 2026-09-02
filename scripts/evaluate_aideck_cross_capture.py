from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flightrl.semantic.aideck_probe_gate import (
    LabeledGray4Capture,
    evaluate_cross_capture_gray4,
)
from flightrl.semantic.aideck_archive import load_archived_frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe an AI Deck scene calibration on distinct labeled captures."
    )
    parser.add_argument("--positive-calibration", type=Path, required=True)
    parser.add_argument("--negative-calibration", type=Path, required=True)
    parser.add_argument("--positive-probe", type=Path, action="append", default=[])
    parser.add_argument("--negative-probe", type=Path, action="append", default=[])
    parser.add_argument("--sample-count", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.sample_count <= 0:
        parser.error("--sample-count must be positive")
    if not args.positive_probe and not args.negative_probe:
        parser.error("at least one distinct probe capture is required")

    positive = _load(args.positive_calibration, "positive", args.sample_count)
    negative = _load(args.negative_calibration, "negative", args.sample_count)
    probes = [
        *(_load(path, "positive", args.sample_count) for path in args.positive_probe),
        *(_load(path, "negative", args.sample_count) for path in args.negative_probe),
    ]
    report = evaluate_cross_capture_gray4(positive, negative, probes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if report["cross_capture_observability_passed"] else 2


def _load(path: Path, label: str, sample_count: int) -> LabeledGray4Capture:
    archived = load_archived_frames(path, sample_count)
    return LabeledGray4Capture(
        label=label,
        frames=np.stack([frame.pixels for frame in archived]),
        indices=tuple(frame.index for frame in archived),
        source=path,
        metadata=archived[0].capture_metadata,
    )


if __name__ == "__main__":
    raise SystemExit(main())
