from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from flightrl.hardware.avoidance_policy import RangerAvoidancePolicy, normalize_reading, reading_from_telemetry
from flightrl.hardware.avoidance_shadow import load_ranger_policy
from flightrl.tracking import add_wandb_args, args_config, init_wandb, log_artifacts, log_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a ranger avoidance checkpoint from Crazyflie command logs")
    parser.add_argument("--input", action="append", required=True, help="Crazyflie CSV with ranger telemetry and vx/vy/yaw/z commands.")
    parser.add_argument("--val-input", action="append", default=[])
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=17)
    add_wandb_args(parser, default_project="FlightRL")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    train_x, train_y = load_examples([Path(path) for path in args.input])
    val_x, val_y = load_examples([Path(path) for path in args.val_input]) if args.val_input else (train_x, train_y)
    model = load_ranger_policy(args.init_checkpoint) if args.init_checkpoint else RangerAvoidancePolicy(hidden_size=args.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    run = init_wandb(args, args_config(args, {"train_samples": int(train_x.shape[0]), "val_samples": int(val_x.shape[0])}))
    history, best_state = train(model, optimizer, train_x, train_y, val_x, val_y, args, run)
    model.load_state_dict(best_state)
    report = build_report(args, history, train_x, val_x)

    output = Path(args.checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_size": int(args.hidden_size),
            "trainer": "ranger_log_imitation",
            "source_logs": args.input,
            "validation_logs": args.val_input,
            "metrics": report["metrics"],
            "note": "Real-log imitation checkpoint; use shadow mode before direct hardware control.",
        },
        output,
    )
    write_report(report, Path(args.report))
    log_metrics(run, {f"final/{key}": value for key, value in report["metrics"].items()})
    log_artifacts(run, name=Path(args.checkpoint).stem, paths=[args.checkpoint, args.report, Path(args.report).with_suffix(".md")], artifact_type="model")
    if run is not None:
        run.finish()
    print(f"checkpoint={output}")
    print(f"report={args.report}")
    print(f"val_loss={report['metrics']['val_loss']:.6f}")


def load_examples(paths: list[Path]) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    for path in paths:
        rows.extend(csv.DictReader(path.open()))
    observations = []
    targets = []
    for row in rows:
        if not has_required_columns(row):
            continue
        telemetry = {key: float(value) for key, value in row.items() if numeric(value)}
        observations.append(normalize_reading(reading_from_telemetry(telemetry)))
        targets.append([float(row["vx_m_s"]), float(row["vy_m_s"]), float(row["yawrate_deg_s"]), float(row["zdistance_m"])])
    if not observations:
        raise SystemExit("no usable ranger command rows found")
    return torch.from_numpy(np.stack(observations)), torch.tensor(targets, dtype=torch.float32)


def train(model, optimizer, train_x, train_y, val_x, val_y, args, run=None) -> tuple[list[dict[str, float]], dict]:
    history = []
    best_val_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(train_x.shape[0])
        losses = []
        for start in range(0, train_x.shape[0], args.batch_size):
            indices = permutation[start : start + args.batch_size]
            prediction = model(train_x[indices])
            loss = F.mse_loss(prediction, train_y[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val_loss = float(F.mse_loss(model(val_x), val_y))
        entry = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss}
        history.append(entry)
        log_metrics(run, {"train/loss": entry["train_loss"], "val/loss": val_loss}, step=epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0:
            print(f"epoch={epoch} train_loss={entry['train_loss']:.6f} val_loss={val_loss:.6f}", flush=True)
    return history, best_state


def build_report(args, history, train_x, val_x) -> dict:
    best = min(history, key=lambda entry: entry["val_loss"])
    latest = history[-1]
    return {
        "checkpoint": args.checkpoint,
        "inputs": args.input,
        "validation_inputs": args.val_input,
        "train_samples": int(train_x.shape[0]),
        "val_samples": int(val_x.shape[0]),
        "metrics": {"best_val_loss": best["val_loss"], "best_epoch": best["epoch"], "val_loss": latest["val_loss"]},
        "history": history,
        "safety": "Not hardware-approved for direct control; validate with shadow logging first.",
    }


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    path.with_suffix(".md").write_text(render_report(report) + "\n")


def render_report(report: dict) -> str:
    metrics = report["metrics"]
    return "\n".join(
        [
            "# Ranger Log Imitation Report",
            "",
            f"- Checkpoint: `{report['checkpoint']}`",
            f"- Train samples: `{report['train_samples']}`",
            f"- Validation samples: `{report['val_samples']}`",
            f"- Best val loss: `{metrics['best_val_loss']:.6f}` at epoch `{metrics['best_epoch']}`",
            f"- Final val loss: `{metrics['val_loss']:.6f}`",
            "",
            report["safety"],
        ]
    )


def has_required_columns(row: dict[str, str]) -> bool:
    return all(numeric(row.get(key, "")) for key in ("range.front", "range.back", "range.left", "range.right", "range.zrange", "vx_m_s", "vy_m_s", "yawrate_deg_s", "zdistance_m"))


def numeric(value: object) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


if __name__ == "__main__":
    main()
