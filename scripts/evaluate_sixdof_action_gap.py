from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flightrl.sixdof import load_policy_from_checkpoint
from flightrl.sixdof.dataset import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate policy action error on an offline 6-DoF teacher dataset")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = load_policy_from_checkpoint(checkpoint)
    dataset = load_dataset(args.dataset)
    observations = dataset["observations"]
    targets = dataset["actions"]
    if int(checkpoint.get("observation_dim", observations.shape[1])) != observations.shape[1]:
        raise SystemExit("checkpoint observation_dim does not match dataset")
    predictions = predict(model, observations, args.batch_size)
    report = build_report(args.checkpoint, args.dataset, dataset["metadata"], predictions, targets, dataset["task_indices"])
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
        print(f"wrote {output}")
    else:
        print(text)


def predict(model, observations: np.ndarray, batch_size: int) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            batch = torch.from_numpy(observations[start : start + batch_size]).float()
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def build_report(checkpoint: str, dataset: str, metadata: dict, predictions: np.ndarray, targets: np.ndarray, task_indices: np.ndarray) -> dict:
    errors = predictions - targets
    l2 = np.linalg.norm(errors, axis=1)
    report = {
        "checkpoint": checkpoint,
        "dataset": dataset,
        "dataset_metadata": metadata,
        "samples": int(len(targets)),
        "mse": float(np.mean(errors * errors)),
        "mae": float(np.mean(np.abs(errors))),
        "l2_mean": float(np.mean(l2)),
        "l2_p95": float(np.quantile(l2, 0.95)),
        "action_saturation_fraction": float(np.mean(np.abs(predictions) > 0.95)),
        "per_task": {},
    }
    tasks = metadata.get("tasks", [])
    for index, task in enumerate(tasks):
        mask = task_indices == index
        if not np.any(mask):
            continue
        task_l2 = l2[mask]
        task_errors = errors[mask]
        report["per_task"][task] = {
            "samples": int(np.sum(mask)),
            "mse": float(np.mean(task_errors * task_errors)),
            "l2_mean": float(np.mean(task_l2)),
            "l2_p95": float(np.quantile(task_l2, 0.95)),
        }
    return report


if __name__ == "__main__":
    main()
