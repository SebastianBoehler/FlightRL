from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def aggregate_training_logs(logs: list[dict]) -> dict:
    episodes = sum(item.get("n", 0.0) for item in logs)
    if episodes == 0:
        return {}
    keys = set().union(*(item.keys() for item in logs)) - {"n"}
    result = {
        key: sum(item.get(key, 0.0) * item.get("n", 0.0) for item in logs)
        / episodes
        for key in keys
    }
    result["n"] = episodes
    return result


def persist_grounder_failure(
    *,
    output_dir: Path,
    env_name: str,
    seed: int,
    state: dict,
    bootstrap: dict,
    metadata: dict,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"{env_name}_seed{seed}_grounder_failed.pt"
    torch.save(state, checkpoint)
    report = {
        "experiment": "D1-door-grounder",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest(),
        "bootstrap": bootstrap,
        "deployment_status": "perception gate failed; no control training",
        **metadata,
    }
    report_path = checkpoint.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return checkpoint, report_path
