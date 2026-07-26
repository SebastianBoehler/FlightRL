from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from flightrl.hardware.direct_raw_gate import DirectRawGateThresholds, evaluate_direct_raw_replay


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate raw Puffer manual-rate replay outputs before live flight.")
    parser.add_argument("--replay", action="append", required=True, help="LABEL:CSV raw-action replay output.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    thresholds = DirectRawGateThresholds()
    reports = {}
    for item in args.replay:
        label, path = split_label_path(item)
        reports[label] = {"path": path, **evaluate_direct_raw_replay(load_rows(path), thresholds)}
    report = {
        "passed": all(item["passed"] for item in reports.values()),
        "reports": reports,
        "safety": "Offline direct-raw replay gate only; passing this report does not approve unsupervised live flight.",
    }
    write_report(report, Path(args.output))
    print(f"direct_raw_gate={args.output}")
    print(f"passed={report['passed']}")
    if args.fail_on_gate and not report["passed"]:
        raise SystemExit(2)


def split_label_path(item: str) -> tuple[str, str]:
    if ":" not in item:
        raise SystemExit("--replay must be LABEL:CSV")
    label, path = item.split(":", 1)
    if not label or not path:
        raise SystemExit("--replay must be LABEL:CSV")
    return label, path


def load_rows(path: str | Path) -> list[dict[str, float]]:
    rows = []
    latest: dict[str, float] = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            latest.update({key: parse_float(value) for key, value in row.items() if value != ""})
            rows.append(dict(latest))
    return rows


def parse_float(raw: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def write_report(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")


def render_markdown(report: dict) -> str:
    lines = ["# Direct Raw Action Gate", "", f"Passed: `{report['passed']}`", ""]
    lines.append("| replay | passed | failures | safe rows | close rows | precontact speed max | sat | thrust p05/p95 | roll/pitch/yaw p95 abs |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
    for label, item in report["reports"].items():
        safe = item.get("safe") or {}
        source = item.get("source") or {}
        rates = f"{safe.get('roll_rate_abs_p95', 0):.1f}/{safe.get('pitch_rate_abs_p95', 0):.1f}/{safe.get('yaw_rate_abs_p95', 0):.1f}"
        thrust = f"{safe.get('thrust_percent_p05', 0):.1f}/{safe.get('thrust_percent_p95', 0):.1f}"
        lines.append(
            f"| {label} | `{item['passed']}` | {', '.join(item['failures']) or 'none'} | "
            f"{item['safe_rows']} | {item['close_safe_rows']} | {source.get('precontact_horizontal_speed_max_m_s', 0):.3f} | "
            f"{safe.get('action_saturation_fraction', 0):.3f} | {thrust} | {rates} |"
        )
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
