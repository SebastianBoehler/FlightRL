from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.replay import assess_log_quality, load_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a Crazyflie replay CSV has enough signal for sim-to-real calibration")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--min-duration-s", type=float, default=5.0)
    parser.add_argument("--min-range-valid-ratio", type=float, default=0.25)
    parser.add_argument("--strict", action="store_true", help="exit non-zero when the log is not calibration-ready")
    args = parser.parse_args()

    report = {
        "input": args.input,
        "quality": assess_log_quality(
            load_rows(args.input),
            min_rows=args.min_rows,
            min_duration_s=args.min_duration_s,
            min_range_valid_ratio=args.min_range_valid_ratio,
        ),
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
        output.with_suffix(".md").write_text(render_markdown(report) + "\n")
        print(f"summary={output}")
        print(f"markdown={output.with_suffix('.md')}")
    else:
        print(text)
    if args.strict and not report["quality"]["calibration_ready"]:
        raise SystemExit(2)


def render_markdown(report: dict) -> str:
    quality = report["quality"]
    lines = [
        "# Replay Log Quality",
        "",
        f"- Input: `{report['input']}`",
        f"- Calibration ready: `{quality['calibration_ready']}`",
        f"- Failures: `{', '.join(quality['failures']) or 'none'}`",
        f"- Rows: `{quality['rows']}`",
        f"- Duration s: `{quality['duration_s']:.3f}`",
        f"- Sample rate Hz: `{quality['sample_rate_hz']:.3f}`",
        f"- Missing columns: `{', '.join(quality['missing_columns']) or 'none'}`",
        "",
        "| range column | valid ratio |",
        "| --- | ---: |",
    ]
    for key, ratio in quality["range_valid_ratio"].items():
        lines.append(f"| {key} | {ratio:.4f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
