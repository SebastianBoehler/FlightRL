from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FlightRL vs Puffer drone native optimization evidence report")
    parser.add_argument("--native-benchmark", type=Path, required=True)
    parser.add_argument("--training-throughput", type=Path, required=True)
    parser.add_argument("--baseline-native-sps", type=float, default=None)
    parser.add_argument("--min-native-sps", type=float, default=1_000_000.0)
    parser.add_argument("--min-total-sps", type=float, default=1_000.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/replay/puffer_native_optimization_report.json"))
    args = parser.parse_args()

    report = build_report(args)
    write_report(report, args.output)
    print(f"optimization_report={args.output}")
    print(f"passed={report['passed']}")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    native = read_json(args.native_benchmark)
    training = read_json(args.training_throughput)
    native_sps = float(native.get("best", {}).get("native_env_steps_per_second", 0.0) or 0.0)
    total_sps = float((training.get("summary", {}).get("best_total_sps") or {}).get("total_sps", 0.0) or 0.0)
    failures = []
    if native_sps < args.min_native_sps:
        failures.append("native_sps_below_threshold")
    if total_sps < args.min_total_sps:
        failures.append("training_sps_below_threshold")
    if args.baseline_native_sps and native_sps < args.baseline_native_sps:
        failures.append("native_sps_regressed")
    return {
        "passed": not failures,
        "failures": failures,
        "comparison": comparison(),
        "throughput": {
            "native_benchmark": str(args.native_benchmark),
            "training_throughput": str(args.training_throughput),
            "native_env_steps_per_second": native_sps,
            "training_total_steps_per_second": total_sps,
            "native_speedup_vs_baseline": native_sps / args.baseline_native_sps if args.baseline_native_sps else None,
            "best_native": native.get("best", {}),
            "best_training": training.get("summary", {}).get("best_total_sps"),
        },
        "safety": "Optimization evidence is sim-only and does not approve live Crazyflie deployment.",
    }


def comparison() -> dict[str, Any]:
    return {
        "puffer_tensaur_drone": {
            "observation_dim": 23,
            "action_interface": "4 direct motor RPM-style commands",
            "integration": "native C RK4 dynamics with motor RPM state",
            "reset_task_handling": "Puffer-native C environment tasks",
        },
        "flightrl": {
            "observation_dim": 28,
            "action_interface": "4 bounded high-level thrust/body-rate commands",
            "integration": "native C batched 6-DoF step with task/reward context",
            "reset_task_handling": "shared native reset helper for Puffer export; Python wrapper exposes deterministic native reset",
        },
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")


def render_markdown(report: dict[str, Any]) -> str:
    throughput = report["throughput"]
    lines = [
        "# Puffer-Native Optimization Report",
        "",
        f"- Passed: `{report['passed']}`",
        f"- Native env SPS: `{throughput['native_env_steps_per_second']:.0f}`",
        f"- Training total SPS: `{throughput['training_total_steps_per_second']:.0f}`",
        f"- Failures: `{', '.join(report['failures']) or 'none'}`",
        "",
        "| system | obs dim | action interface | integration |",
        "| --- | ---: | --- | --- |",
    ]
    for name, row in report["comparison"].items():
        lines.append(f"| {name} | {row['observation_dim']} | {row['action_interface']} | {row['integration']} |")
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


if __name__ == "__main__":
    main()
