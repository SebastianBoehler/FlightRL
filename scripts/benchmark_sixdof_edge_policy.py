from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from flightrl.edge import sample_sixdof_observations
from flightrl.sixdof import load_policy_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark 6-DoF checkpoint and optional TorchScript edge inference latency")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--torchscript", default=None)
    parser.add_argument("--output", default="artifacts/edge/sixdof_latency.json")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    inputs = sample_inputs(checkpoint, args.batch_size, args.seed)
    eager = load_policy_from_checkpoint(checkpoint).net
    eager.eval()
    report = {
        "checkpoint": args.checkpoint,
        "torchscript": args.torchscript,
        "batch_size": args.batch_size,
        "iterations": args.iterations,
        "observation": {
            "shape": [int(checkpoint.get("observation_dim", 28))],
            "mode": checkpoint.get("observation_mode", "base"),
        },
        "eager": benchmark_module(eager, inputs, args.iterations, args.warmup),
        "safety": "Local inference benchmark only; not approved for live hardware.",
    }
    if args.torchscript:
        scripted = torch.jit.load(args.torchscript, map_location="cpu")
        scripted.eval()
        report["torchscript_result"] = benchmark_module(scripted, inputs, args.iterations, args.warmup)
        with torch.no_grad():
            report["max_abs_error"] = float(torch.max(torch.abs(eager(inputs) - scripted(inputs))).item())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
    print(f"latency={output}")
    print(f"markdown={output.with_suffix('.md')}")


def sample_inputs(checkpoint: dict, batch_size: int, seed: int) -> torch.Tensor:
    task_count = len(checkpoint.get("tasks", [])) if checkpoint.get("task_conditioned", False) else 1
    return sample_sixdof_observations(
        seed=seed,
        samples=batch_size,
        observation_dim=int(checkpoint.get("observation_dim", 28)),
        observation_mode=str(checkpoint.get("observation_mode", "base")),
        task_count=task_count,
    )


def benchmark_module(module, inputs: torch.Tensor, iterations: int, warmup: int) -> dict:
    with torch.no_grad():
        for _ in range(warmup):
            module(inputs)
        start = perf_counter()
        for _ in range(iterations):
            module(inputs)
        elapsed = perf_counter() - start
    per_batch_us = elapsed * 1_000_000.0 / max(iterations, 1)
    per_sample_us = per_batch_us / max(int(inputs.shape[0]), 1)
    return {
        "elapsed_s": elapsed,
        "per_batch_us": per_batch_us,
        "per_sample_us": per_sample_us,
        "samples_per_second": 1_000_000.0 / per_sample_us,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# 6-DoF Edge Latency",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Observation mode: `{report['observation']['mode']}`",
        f"- Batch size: `{report['batch_size']}`",
        f"- Eager per-sample us: `{report['eager']['per_sample_us']:.3f}`",
    ]
    if "torchscript_result" in report:
        lines.append(f"- TorchScript per-sample us: `{report['torchscript_result']['per_sample_us']:.3f}`")
        lines.append(f"- Max abs error: `{report['max_abs_error']:.8f}`")
    lines.extend(["", report["safety"]])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
