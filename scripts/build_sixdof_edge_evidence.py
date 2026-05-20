from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or run TorchScript edge evidence generation for candidate-matrix checkpoints")
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--include-best-by-task", action="store_true")
    parser.add_argument("--include-best-multitask", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/edge")
    parser.add_argument("--report", default="artifacts/replay/sixdof_edge_evidence_manifest.json")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()

    matrix = json.loads(Path(args.matrix).read_text())
    selected = select_records(matrix, args)
    records = [evidence_record(record, args) for record in selected]
    if args.run:
        for record in records:
            record["results"] = run_commands(record["commands"])
    report = {"run": args.run, "matrix": args.matrix, "records": records, "matrix_args": matrix_args(records)}
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
    print(f"edge_evidence={output}")
    print(f"markdown={output.with_suffix('.md')}")


def select_records(matrix: dict, args: argparse.Namespace) -> list[dict]:
    selected = []
    if args.include_best_by_task:
        selected.extend(matrix.get("best_by_task", {}).values())
    if args.include_best_multitask and matrix.get("best_multitask"):
        selected.append(matrix["best_multitask"])
    by_label = {record["label"]: record for record in matrix.get("records", [])}
    missing = [label for label in args.label if label not in by_label]
    if missing:
        raise SystemExit(f"Unknown label(s): {', '.join(missing)}")
    selected.extend(by_label[label] for label in args.label)
    return dedupe(selected)


def dedupe(records: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for record in records:
        key = (record["label"], record["checkpoint"])
        if key not in seen:
            unique.append(record)
            seen.add(key)
    return unique


def evidence_record(record: dict, args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    stem = safe_stem(record["label"])
    model = output_dir / f"{stem}.ts"
    parity = output_dir / f"{stem}.parity.json"
    latency = output_dir / f"{stem}.latency.json"
    return {
        "label": record["label"],
        "checkpoint": record["checkpoint"],
        "tasks": record.get("tasks", []),
        "model": str(model),
        "parity": str(parity),
        "latency": str(latency),
        "commands": [
            export_command(record["checkpoint"], model, parity, args.samples),
            latency_command(record["checkpoint"], model, latency, args.iterations, args.warmup),
        ],
    }


def export_command(checkpoint: str, model: Path, parity: Path, samples: int) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "export_sixdof_edge_policy.py"),
        "--checkpoint",
        checkpoint,
        "--output",
        str(model),
        "--report",
        str(parity),
        "--samples",
        str(samples),
    ]


def latency_command(checkpoint: str, model: Path, latency: Path, iterations: int, warmup: int) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "benchmark_sixdof_edge_policy.py"),
        "--checkpoint",
        checkpoint,
        "--torchscript",
        str(model),
        "--iterations",
        str(iterations),
        "--warmup",
        str(warmup),
        "--output",
        str(latency),
    ]


def run_commands(commands: list[list[str]]) -> list[dict]:
    results = []
    for command in commands:
        start = perf_counter()
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        results.append(
            {
                "command": command,
                "returncode": completed.returncode,
                "elapsed_s": perf_counter() - start,
                "stderr_tail": completed.stderr.splitlines()[-5:],
            }
        )
        if completed.returncode != 0:
            break
    return results


def matrix_args(records: list[dict]) -> list[str]:
    args = []
    for record in records:
        args.extend(
            [
                "--parity",
                f"{record['label']}={record['parity']}",
                "--latency",
                f"{record['label']}={record['latency']}",
            ]
        )
    return args


def render_markdown(report: dict) -> str:
    lines = ["# 6-DoF Edge Evidence Manifest", "", "| label | tasks | status | parity | latency |", "| --- | --- | --- | --- | --- |"]
    for record in report["records"]:
        lines.append(f"| {record['label']} | {', '.join(record['tasks'])} | {status(record)} | `{record['parity']}` | `{record['latency']}` |")
    lines.extend(["", "Append `matrix_args` from the JSON report to `build_sixdof_candidate_matrix.py`."])
    return "\n".join(lines)


def status(record: dict) -> str:
    results = record.get("results")
    if not results:
        return "planned"
    return "ok" if all(result["returncode"] == 0 for result in results) else "failed"


def safe_stem(label: str) -> str:
    return "sixdof_" + "".join(char if char.isalnum() or char in "-_" else "_" for char in label)


if __name__ == "__main__":
    main()
