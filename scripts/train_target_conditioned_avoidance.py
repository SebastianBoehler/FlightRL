from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from flightrl.hardware.avoidance_policy import RangerReading, command_array, reading_from_telemetry, sample_readings
from flightrl.hardware.target_direction import TargetDirectionConfig, target_direction_command
from flightrl.hardware.target_conditioned_policy import TargetConditionedPolicy, TargetSpec, load_target_policy, target_observation


def main() -> None:
    parser = argparse.ArgumentParser(description="Train target-conditioned ranger avoidance from directional Crazyflie logs")
    parser.add_argument("--input", action="append", required=True, metavar="CSV,DIRECTION_DEG,SPEED_M_S")
    parser.add_argument("--val-input", action="append", default=[], metavar="CSV,DIRECTION_DEG,SPEED_M_S")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--synthetic-samples", type=int, default=0)
    parser.add_argument("--synthetic-close-samples", type=int, default=0)
    parser.add_argument("--synthetic-speed-m-s", type=float, default=0.16)
    parser.add_argument("--synthetic-max-speed-m-s", type=float, default=0.55)
    parser.add_argument("--synthetic-close-min-m", type=float, default=0.05)
    parser.add_argument("--synthetic-close-max-m", type=float, default=0.55)
    parser.add_argument("--clearance-m", type=float, default=1.30)
    parser.add_argument("--hard-clearance-m", type=float, default=0.10)
    parser.add_argument("--target-height-m", type=float, default=0.50)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    train_x, train_y = load_examples(args.input)
    if args.synthetic_samples:
        synthetic_x, synthetic_y = synthetic_examples(args, rng)
        train_x = torch.cat([train_x, synthetic_x], dim=0)
        train_y = torch.cat([train_y, synthetic_y], dim=0)
    if args.synthetic_close_samples:
        close_x, close_y = synthetic_close_examples(args, rng)
        train_x = torch.cat([train_x, close_x], dim=0)
        train_y = torch.cat([train_y, close_y], dim=0)
    val_x, val_y = load_examples(args.val_input) if args.val_input else (train_x, train_y)
    model = load_target_policy(args.init_checkpoint) if args.init_checkpoint else TargetConditionedPolicy(hidden_size=args.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history, best_state = train(model, optimizer, train_x, train_y, val_x, val_y, args)
    model.load_state_dict(best_state)
    report = build_report(args, history, train_x, val_x)

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_size": args.hidden_size,
            "trainer": "target_conditioned_log_imitation",
            "source_logs": args.input,
            "validation_logs": args.val_input,
            "metrics": report["metrics"],
            "note": "Target-conditioned imitation checkpoint; use shadow mode before direct hardware control.",
        },
        checkpoint,
    )
    write_report(report, Path(args.report))
    print(f"checkpoint={checkpoint}")
    print(f"report={args.report}")
    print(f"best_val_loss={report['metrics']['best_val_loss']:.6f}")


def load_examples(specs: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    observations = []
    targets = []
    for spec in specs:
        path, target = parse_input_spec(spec)
        for row in csv.DictReader(path.open()):
            if not has_required_columns(row):
                continue
            telemetry = {key: float(value) for key, value in row.items() if numeric(value)}
            observations.append(target_observation(reading_from_telemetry(telemetry), target))
            targets.append([float(row["vx_m_s"]), float(row["vy_m_s"]), float(row["yawrate_deg_s"]), float(row["zdistance_m"])])
    if not observations:
        raise SystemExit("no usable target-direction command rows found")
    return torch.from_numpy(np.stack(observations)), torch.tensor(targets, dtype=torch.float32)


def synthetic_examples(args, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    readings = sample_readings(args.synthetic_samples, rng)
    return examples_for_readings(args, readings, rng)


def synthetic_close_examples(args, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    readings = []
    sides = rng.integers(0, 4, size=args.synthetic_close_samples)
    for side in sides:
        values = rng.uniform(0.7, 3.2, size=6)
        values[side] = rng.uniform(args.synthetic_close_min_m, args.synthetic_close_max_m)
        values[5] = rng.uniform(0.42, 0.56)
        readings.append(RangerReading(*[float(value) for value in values]))
    return examples_for_readings(args, readings, rng)


def examples_for_readings(args, readings, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    observations = []
    targets = []
    directions = rng.uniform(0.0, 360.0, size=len(readings))

    for reading, direction_deg in zip(readings, directions, strict=True):
        target = TargetSpec(float(direction_deg), args.synthetic_speed_m_s)
        command = target_direction_command(
            reading,
            TargetDirectionConfig(
                direction_deg=target.direction_deg,
                target_speed_m_s=target.speed_m_s,
                clearance_m=args.clearance_m,
                hard_clearance_m=args.hard_clearance_m,
                target_height_m=args.target_height_m,
                avoidance_speed_m_s=args.synthetic_max_speed_m_s,
                max_speed_m_s=args.synthetic_max_speed_m_s,
            ),
        )
        observations.append(target_observation(reading, target))
        targets.append(command_array(command))
    return torch.from_numpy(np.stack(observations)), torch.from_numpy(np.stack(targets))


def parse_input_spec(spec: str) -> tuple[Path, TargetSpec]:
    pieces = spec.rsplit(",", 2)
    if len(pieces) != 3:
        raise SystemExit(f"input spec must be CSV,DIRECTION_DEG,SPEED_M_S: {spec}")
    return Path(pieces[0]), TargetSpec(direction_deg=float(pieces[1]), speed_m_s=float(pieces[2]))


def train(model, optimizer, train_x, train_y, val_x, val_y, args) -> tuple[list[dict[str, float]], dict]:
    history = []
    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        permutation = torch.randperm(train_x.shape[0])
        for start in range(0, train_x.shape[0], args.batch_size):
            indices = permutation[start : start + args.batch_size]
            loss = F.mse_loss(model(train_x[indices]), train_y[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val_loss = float(F.mse_loss(model(val_x), val_y))
        entry = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss}
        history.append(entry)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0:
            print(f"epoch={epoch} train_loss={entry['train_loss']:.6f} val_loss={val_loss:.6f}", flush=True)
    return history, best_state


def build_report(args, history, train_x, val_x) -> dict:
    best = min(history, key=lambda entry: entry["val_loss"])
    return {
        "checkpoint": args.checkpoint,
        "inputs": args.input,
        "validation_inputs": args.val_input,
        "train_samples": int(train_x.shape[0]),
        "val_samples": int(val_x.shape[0]),
        "metrics": {"best_val_loss": best["val_loss"], "best_epoch": best["epoch"], "final_val_loss": history[-1]["val_loss"]},
        "safety": "Not hardware-approved for direct control; validate with target-conditioned shadow logging first.",
    }


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def has_required_columns(row: dict[str, str]) -> bool:
    required = ("range.front", "range.back", "range.left", "range.right", "range.zrange", "vx_m_s", "vy_m_s", "yawrate_deg_s", "zdistance_m")
    return all(numeric(row.get(key, "")) for key in required)


def numeric(value: object) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


if __name__ == "__main__":
    main()
