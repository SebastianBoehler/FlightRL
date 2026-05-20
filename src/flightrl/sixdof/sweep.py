from __future__ import annotations

import json
from pathlib import Path
import subprocess
from time import perf_counter


def run_commands(commands: list[list[str]], *, cwd: Path) -> list[dict]:
    results = []
    for command in commands:
        start = perf_counter()
        completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
        results.append({"command": command, "returncode": completed.returncode, "elapsed_s": perf_counter() - start})
        if completed.returncode != 0:
            break
    return results


def load_suite_summary(path: str) -> dict | None:
    suite = Path(path)
    if not suite.exists():
        return None
    record = json.loads(suite.read_text())["records"][0]
    metrics = record["metrics"]
    return {
        "passed": record["gate"]["passed"],
        "failures": record["gate"]["failures"],
        "mean_completed_fraction": metrics["mean_completed_fraction"],
        "mean_survival_fraction": metrics["mean_survival_fraction"],
        "mean_position_error_m": metrics["mean_position_error_m"],
        "clearance_p01_m": metrics["clearance_p01_m"],
        "per_task_gate": record.get("per_task_gate", {}),
    }


def sweep_summary(records: list[dict]) -> dict:
    return {"total": len(records), "completed": sum(1 for record in records if all_success(record.get("results"))), "best": best_record(records)}


def all_success(results: list[dict] | None) -> bool:
    return bool(results) and all(result["returncode"] == 0 for result in results)


def best_record(records: list[dict]) -> dict | None:
    candidates = [(gate_score(record["gate"]), compact_record(record)) for record in records if record.get("gate")]
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def gate_score(gate: dict) -> tuple:
    return (
        0 if gate["passed"] else 1,
        -gate["mean_completed_fraction"],
        -gate["mean_survival_fraction"],
        gate["mean_position_error_m"],
        -gate["clearance_p01_m"],
    )


def compact_record(record: dict) -> dict:
    gate = record["gate"]
    return {"name": record["variant"]["name"], "checkpoint": record["checkpoint"], **gate}


def status(record: dict) -> str:
    if "results" not in record:
        return "planned"
    return "ok" if all_success(record["results"]) else "failed"


def fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "pending"
