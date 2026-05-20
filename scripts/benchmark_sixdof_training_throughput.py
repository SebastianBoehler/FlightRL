from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter

import torch

from flightrl.sixdof import SixDofCrazyflieEnv
from flightrl.sixdof.observation import OBSERVATION_MODES, observation_dim
from flightrl.sixdof.rl import PpoConfig, SixDofActorCritic, collect_rollout, ppo_update


@dataclass(slots=True)
class ThroughputVariant:
    name: str
    num_envs: int
    horizon: int
    hidden_size: int
    minibatch_size: int
    update_epochs: int = 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark 6-DoF rollout collection and PPO update throughput")
    parser.add_argument("--output", default="artifacts/replay/sixdof_training_throughput.json")
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument("--task", default="position_yaw")
    parser.add_argument("--reset-profile", default="position_yaw_medium")
    parser.add_argument("--observation-mode", default="base", choices=OBSERVATION_MODES)
    parser.add_argument("--native-step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    variants = select_variants(default_variants(), args.variants)
    if args.max_variants is not None:
        variants = variants[: args.max_variants]
    records = [benchmark_variant(variant, args) for variant in variants]
    report = {"native_step": args.native_step, "observation_mode": args.observation_mode, "records": records, "summary": summarize(records)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
    print(f"summary={output}")
    print(f"markdown={output.with_suffix('.md')}")


def default_variants() -> list[ThroughputVariant]:
    return [
        ThroughputVariant("smoke_64x16_h64", 64, 16, 64, 1024),
        ThroughputVariant("base_256x32_h128", 256, 32, 128, 4096),
        ThroughputVariant("wide_256x32_h256", 256, 32, 256, 4096),
        ThroughputVariant("large_512x32_h128", 512, 32, 128, 8192),
        ThroughputVariant("long_256x64_h128", 256, 64, 128, 8192),
    ]


def select_variants(variants: list[ThroughputVariant], names: list[str] | None) -> list[ThroughputVariant]:
    if not names:
        return variants
    by_name = {variant.name: variant for variant in variants}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise SystemExit(f"Unknown variant(s): {', '.join(missing)}")
    return [by_name[name] for name in names]


def benchmark_variant(variant: ThroughputVariant, args: argparse.Namespace) -> dict:
    torch.manual_seed(17)
    env = SixDofCrazyflieEnv(num_envs=variant.num_envs, seed=17, task=args.task, use_native_step=args.native_step, reset_profile=args.reset_profile)
    model = SixDofActorCritic(input_dim=observation_dim(28, args.observation_mode), hidden_size=variant.hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    config = PpoConfig(hidden_size=variant.hidden_size, minibatch_size=variant.minibatch_size, update_epochs=variant.update_epochs)
    if args.warmup:
        ppo_update(model, optimizer, collect_rollout(env, model, horizon=2, action_std=config.action_std, observation_mode=args.observation_mode), config)
    collect_start = perf_counter()
    rollout = collect_rollout(env, model, horizon=variant.horizon, action_std=config.action_std, observation_mode=args.observation_mode)
    collect_s = perf_counter() - collect_start
    update_start = perf_counter()
    losses = ppo_update(model, optimizer, rollout, config)
    update_s = perf_counter() - update_start
    samples = variant.num_envs * variant.horizon
    return {
        "variant": asdict(variant),
        "samples": samples,
        "collect_s": collect_s,
        "update_s": update_s,
        "total_s": collect_s + update_s,
        "collect_sps": samples / collect_s,
        "update_sps": samples / update_s,
        "total_sps": samples / (collect_s + update_s),
        "losses": losses,
    }


def summarize(records: list[dict]) -> dict:
    return {
        "total": len(records),
        "best_collect_sps": compact(max(records, key=lambda row: row["collect_sps"], default=None), "collect_sps"),
        "best_total_sps": compact(max(records, key=lambda row: row["total_sps"], default=None), "total_sps"),
    }


def compact(record: dict | None, metric: str) -> dict | None:
    if record is None:
        return None
    variant = record["variant"]
    return {"name": variant["name"], metric: record[metric], "num_envs": variant["num_envs"], "horizon": variant["horizon"], "hidden_size": variant["hidden_size"]}


def render_markdown(report: dict) -> str:
    lines = ["# 6-DoF Training Throughput", "", "| variant | envs | horizon | hidden | collect SPS | update SPS | total SPS |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for record in report["records"]:
        variant = record["variant"]
        lines.append(
            f"| {variant['name']} | {variant['num_envs']} | {variant['horizon']} | {variant['hidden_size']} | "
            f"{record['collect_sps']:.0f} | {record['update_sps']:.0f} | {record['total_sps']:.0f} |"
        )
    best = report["summary"].get("best_total_sps")
    if best:
        lines.extend(["", f"Best total throughput: `{best['name']}` at `{best['total_sps']:.0f}` samples/sec."])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
