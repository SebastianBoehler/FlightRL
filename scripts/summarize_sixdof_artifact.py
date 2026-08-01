from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.evidence_scope import DESKTOP_DEVELOPMENT_SCOPE


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a 6-DoF simulation gate and desktop parity artifacts"
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--desktop-parity", default=None)
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
    parity = read_json(args.desktop_parity) if args.desktop_parity else None
    metrics = gate["metrics"]
    return {
        "evidence_scope": DESKTOP_DEVELOPMENT_SCOPE,
        "deployment_authority": False,
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
        "desktop_parity": parity,
        "safety": "Simulation and desktop CPU summary only; not AI Deck deployment readiness or live-hardware authority.",
    }


def read_json(path: str | None) -> dict | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text())


def render_markdown(summary: dict) -> str:
    gate = summary["gate"]
    metrics = summary["gate_metrics"]
    parity = summary["desktop_parity"]
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
        "",
        summary["safety"],
    ]
    if parity:
        lines.extend(
            [
                "",
                "## Desktop TorchScript Parity",
                "",
                f"- Model: `{parity['model']}`",
                f"- Max abs error: `{parity['parity']['max_abs_error']:.8f}`",
                f"- Mean abs error: `{parity['parity']['mean_abs_error']:.8f}`",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
