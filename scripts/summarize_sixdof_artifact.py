from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a 6-DoF policy checkpoint, gate, gap, and edge parity artifacts")
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--action-gap", required=True)
    parser.add_argument("--edge-parity", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary = build_summary(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    markdown = output.with_suffix(".md")
    markdown.write_text(render_markdown(summary) + "\n")
    print(f"summary={output}")
    print(f"markdown={markdown}")


def build_summary(args) -> dict:
    gate = read_json(args.gate)
    gap = read_json(args.action_gap)
    parity = read_json(args.edge_parity) if args.edge_parity else None
    metrics = gate["metrics"]
    return {
        "name": args.name,
        "checkpoint": args.checkpoint,
        "tasks": gate["tasks"],
        "gate": gate["gate"],
        "gate_metrics": {
            "mean_position_error_m": metrics["mean_position_error_m"],
            "mean_completed_fraction": metrics["mean_completed_fraction"],
            "clearance_p01_m": metrics.get("clearance_p01_m", metrics["min_clearance_m"]),
            "action_saturation_fraction": metrics.get("action_saturation_fraction"),
            "teacher_action_l2_mean": metrics.get("teacher_action_l2_mean"),
            "teacher_action_l2_p95": metrics.get("teacher_action_l2_p95"),
        },
        "action_gap": {
            "dataset": gap["dataset"],
            "samples": gap["samples"],
            "l2_mean": gap["l2_mean"],
            "l2_p95": gap["l2_p95"],
            "action_saturation_fraction": gap["action_saturation_fraction"],
            "per_task": gap["per_task"],
        },
        "edge_parity": parity,
        "safety": "Simulation summary only; not approved for direct hardware control.",
    }


def read_json(path: str | None) -> dict | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text())


def render_markdown(summary: dict) -> str:
    gate = summary["gate"]
    metrics = summary["gate_metrics"]
    parity = summary["edge_parity"]
    lines = [
        f"# {summary['name']}",
        "",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Tasks: `{', '.join(summary['tasks'])}`",
        f"- Gate passed: `{gate['passed']}`",
        f"- Gate failures: `{', '.join(gate['failures']) or 'none'}`",
        f"- Completion: `{metrics['mean_completed_fraction']:.4f}`",
        f"- Position error m: `{metrics['mean_position_error_m']:.4f}`",
        f"- Clearance p01 m: `{metrics['clearance_p01_m']:.4f}`",
        f"- Action saturation: `{metrics['action_saturation_fraction']:.6f}`",
        f"- Teacher action L2 mean: `{metrics['teacher_action_l2_mean']:.6f}`",
        f"- Teacher action L2 p95: `{metrics['teacher_action_l2_p95']:.6f}`",
        f"- Action-gap samples: `{summary['action_gap']['samples']}`",
        f"- Action-gap L2 mean/p95: `{summary['action_gap']['l2_mean']:.6f}` / `{summary['action_gap']['l2_p95']:.6f}`",
        "",
        summary["safety"],
    ]
    if parity:
        lines.extend(
            [
                "",
                "## Edge Parity",
                "",
                f"- Model: `{parity['model']}`",
                f"- Max abs error: `{parity['parity']['max_abs_error']:.8f}`",
                f"- Mean abs error: `{parity['parity']['mean_abs_error']:.8f}`",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
