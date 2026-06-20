from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from flightrl.hardware.avoidance_policy import RangerReading, command_array, reactive_clearance_command, reading_from_telemetry
from flightrl.hardware.ttc_policy import TTCAvoidancePolicy, rate_from_telemetry, ttc_observation
from flightrl.tracking import add_wandb_args, args_config, init_wandb, log_artifacts, log_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a TTC/range-rate avoidance checkpoint from Crazyflie logs")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--val-input", action="append", default=[])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--close-weight", type=float, default=2.0)
    parser.add_argument("--relabel-with-teacher", action="store_true")
    parser.add_argument("--label-command", choices=("held", "raw"), default="held")
    parser.add_argument("--clearance-m", type=float, default=0.45)
    parser.add_argument("--hard-clearance-m", type=float, default=0.08)
    parser.add_argument("--height-m", type=float, default=0.50)
    parser.add_argument("--max-speed-m-s", type=float, default=0.65)
    parser.add_argument("--ttc-horizon-s", type=float, default=0.65)
    parser.add_argument("--ttc-hard-s", type=float, default=0.15)
    parser.add_argument("--ttc-gain", type=float, default=0.75)
    parser.add_argument("--synthetic-close-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=23)
    add_wandb_args(parser, default_project="FlightRL")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    train_x, train_y, train_w = load_examples([Path(path) for path in args.input], args)
    if args.synthetic_close_samples:
        train_x, train_y, train_w = append_synthetic_close_examples(train_x, train_y, train_w, args)
    val_x, val_y, val_w = load_examples([Path(path) for path in args.val_input], args) if args.val_input else (train_x, train_y, train_w)
    model = TTCAvoidancePolicy(hidden_size=args.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    run = init_wandb(args, wandb_config(args, train_x, val_x))
    history, best_state = train(model, optimizer, train_x, train_y, train_w, val_x, val_y, val_w, args, run)
    model.load_state_dict(best_state)
    report = build_report(args, history, model, train_x, train_y, val_x, val_y)
    write_outputs(model, report, args)
    log_metrics(run, {f"final/{key}": value for key, value in report["metrics"].items()})
    log_artifacts(run, name=Path(args.checkpoint).stem, paths=[args.checkpoint, args.report, Path(args.report).with_suffix(".md")], artifact_type="model")
    if run is not None:
        run.finish()
    print(f"checkpoint={args.checkpoint}")
    print(f"report={args.report}")
    print(f"val_loss={report['metrics']['val_loss']:.6f}")


def load_examples(paths: list[Path], args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    observations: list[np.ndarray] = []
    targets: list[list[float]] = []
    weights: list[float] = []
    for path in paths:
        for row in csv.DictReader(path.open()):
            if not has_required_columns(row, args):
                continue
            telemetry = {key: float(value) for key, value in row.items() if numeric(value)}
            reading = reading_from_telemetry(telemetry)
            rate = rate_from_telemetry(telemetry)
            observations.append(ttc_observation(reading, rate))
            if args.relabel_with_teacher:
                targets.append(command_array(teacher_label(reading, rate, args)).tolist())
            else:
                targets.append([float(row[key]) for key in label_columns(args)])
            close = float(telemetry.get("min_horizontal_range_m", min(reading.front_m, reading.back_m, reading.left_m, reading.right_m))) < 0.45
            urgent = float(telemetry.get("min_horizontal_ttc_s", 99.0)) < 0.7
            weights.append(float(args.close_weight if close or urgent else 1.0))
    if not observations:
        raise SystemExit("no usable TTC command rows found")
    return (
        torch.from_numpy(np.stack(observations)),
        torch.tensor(targets, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )


def append_synthetic_close_examples(train_x: torch.Tensor, train_y: torch.Tensor, train_w: torch.Tensor, args):
    rng = np.random.default_rng(args.seed)
    observations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[float] = []
    for index in range(args.synthetic_close_samples):
        reading, rate = synthetic_close_state(rng, index, args.height_m)
        command = teacher_label(reading, rate, args)
        observations.append(ttc_observation(reading, rate))
        targets.append(command_array(command))
        weights.append(float(args.close_weight))
    return (
        torch.cat([train_x, torch.from_numpy(np.stack(observations))]),
        torch.cat([train_y, torch.from_numpy(np.stack(targets)).float()]),
        torch.cat([train_w, torch.tensor(weights, dtype=torch.float32)]),
    )


def synthetic_close_state(rng: np.random.Generator, index: int, height_m: float) -> tuple[RangerReading, RangerReading]:
    horizontal = rng.uniform(0.8, 3.0, size=4)
    sides = rng.choice(4, size=1 + (index % 3 == 0), replace=False)
    for side in sides:
        horizontal[side] = rng.uniform(0.08, 0.42)
    rates = np.zeros(4, dtype=np.float32)
    for side in sides:
        rates[side] = -rng.uniform(0.2, 2.5)
    reading = RangerReading(
        front_m=float(horizontal[0]),
        back_m=float(horizontal[1]),
        left_m=float(horizontal[2]),
        right_m=float(horizontal[3]),
        up_m=float(rng.uniform(1.5, 3.5)),
        zrange_m=float(rng.uniform(max(0.25, height_m - 0.08), height_m + 0.08)),
    )
    rate = RangerReading(float(rates[0]), float(rates[1]), float(rates[2]), float(rates[3]), 0.0, 0.0)
    return reading, rate


def teacher_label(reading: RangerReading, rate: RangerReading, args):
    return reactive_clearance_command(
        reading,
        range_rate_m_s=rate,
        clearance_m=args.clearance_m,
        hard_clearance_m=args.hard_clearance_m,
        target_height_m=args.height_m,
        max_speed_m_s=args.max_speed_m_s,
        ttc_horizon_s=args.ttc_horizon_s,
        ttc_hard_s=args.ttc_hard_s,
        ttc_gain=args.ttc_gain,
    )


def train(model, optimizer, train_x, train_y, train_w, val_x, val_y, val_w, args, run=None) -> tuple[list[dict[str, float]], dict]:
    history = []
    best_val_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for indices in batches(torch.randperm(train_x.shape[0]), args.batch_size):
            prediction = model(train_x[indices])
            loss = weighted_mse(prediction, train_y[indices], train_w[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val_loss = float(weighted_mse(model(val_x), val_y, val_w))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        log_metrics(run, {"train/loss": history[-1]["train_loss"], "val/loss": val_loss}, step=epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0:
            print(f"epoch={epoch} train_loss={history[-1]['train_loss']:.6f} val_loss={val_loss:.6f}", flush=True)
    return history, best_state


def wandb_config(args, train_x: torch.Tensor, val_x: torch.Tensor) -> dict:
    return args_config(args, {"train_samples": int(train_x.shape[0]), "val_samples": int(val_x.shape[0])})


def build_report(args, history, model, train_x, train_y, val_x, val_y) -> dict:
    best = min(history, key=lambda entry: entry["val_loss"])
    metrics = {"best_val_loss": best["val_loss"], "best_epoch": best["epoch"], "val_loss": history[-1]["val_loss"]}
    metrics.update(evaluate(model, val_x, val_y, prefix="val"))
    return {
        "checkpoint": args.checkpoint,
        "inputs": args.input,
        "validation_inputs": args.val_input,
        "train_samples": int(train_x.shape[0]),
        "val_samples": int(val_x.shape[0]),
        "label_source": "teacher" if args.relabel_with_teacher else f"log_{args.label_command}",
        "synthetic_close_samples": int(args.synthetic_close_samples),
        "metrics": metrics,
        "history": history,
        "safety": "Supervised TTC imitation; shadow before direct hardware control.",
    }


def evaluate(model, x, y, *, prefix: str) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        prediction = model(x).cpu().numpy()
    actual = y.cpu().numpy()
    error = prediction - actual
    speed = np.linalg.norm(prediction[:, :2], axis=1)
    return {
        f"{prefix}_mae_vx_m_s": float(np.mean(np.abs(error[:, 0]))),
        f"{prefix}_mae_vy_m_s": float(np.mean(np.abs(error[:, 1]))),
        f"{prefix}_mae_zdistance_m": float(np.mean(np.abs(error[:, 3]))),
        f"{prefix}_speed_p95_m_s": float(np.percentile(speed, 95)),
        f"{prefix}_sign_agreement_vx": sign_agreement(actual[:, 0], prediction[:, 0]),
        f"{prefix}_sign_agreement_vy": sign_agreement(actual[:, 1], prediction[:, 1]),
    }


def write_outputs(model, report: dict, args) -> None:
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hidden_size": args.hidden_size, "trainer": "ttc_log_imitation", "metrics": report["metrics"]}, checkpoint)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    report_path.with_suffix(".md").write_text(render_report(report) + "\n")


def render_report(report: dict) -> str:
    metrics = report["metrics"]
    return "\n".join(
        [
            "# TTC Log Imitation Report",
            "",
            f"- Checkpoint: `{report['checkpoint']}`",
            f"- Train samples: `{report['train_samples']}`",
            f"- Validation samples: `{report['val_samples']}`",
            f"- Label source: `{report['label_source']}`",
            f"- Synthetic close samples: `{report['synthetic_close_samples']}`",
            f"- Best val loss: `{metrics['best_val_loss']:.6f}` at epoch `{metrics['best_epoch']}`",
            f"- Val MAE vx/vy/z: `{metrics['val_mae_vx_m_s']:.4f}`, `{metrics['val_mae_vy_m_s']:.4f}`, `{metrics['val_mae_zdistance_m']:.4f}`",
            f"- Val sign agreement vx/vy: `{metrics['val_sign_agreement_vx']:.3f}` / `{metrics['val_sign_agreement_vy']:.3f}`",
            "",
            report["safety"],
        ]
    )


def weighted_mse(prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.mean((prediction - target) ** 2, dim=1) * weight)


def batches(indices: torch.Tensor, batch_size: int):
    for start in range(0, indices.shape[0], batch_size):
        yield indices[start : start + batch_size]


def has_required_columns(row: dict[str, str], args) -> bool:
    required = (
        "range.front",
        "range.back",
        "range.left",
        "range.right",
        "range.zrange",
        "range_rate_front_m_s",
        "range_rate_back_m_s",
        "range_rate_left_m_s",
        "range_rate_right_m_s",
        *(() if args.relabel_with_teacher else label_columns(args)),
    )
    return all(numeric(row.get(key, "")) for key in required)


def label_columns(args) -> tuple[str, str, str, str]:
    if args.label_command == "raw":
        return ("raw_vx_m_s", "raw_vy_m_s", "raw_yawrate_deg_s", "raw_zdistance_m")
    return ("vx_m_s", "vy_m_s", "yawrate_deg_s", "zdistance_m")


def numeric(value: object) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def sign_agreement(actual: np.ndarray, predicted: np.ndarray, *, threshold: float = 0.03) -> float:
    mask = np.abs(actual) >= threshold
    if not np.any(mask):
        return 1.0
    return float(np.mean(np.sign(actual[mask]) == np.sign(predicted[mask])))


if __name__ == "__main__":
    main()
