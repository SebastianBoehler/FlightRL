from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
SPS_RE = re.compile(r"\bSPS\s+([0-9]+(?:\.[0-9]+)?)([KMG]?)\b")


@dataclass(slots=True)
class SweepVariant:
    name: str
    total_agents: int
    num_buffers: int
    num_threads: int
    horizon: int
    minibatch_size: int
    replay_ratio: int
    learning_rate: float
    ent_coef: float
    policy_hidden_size: int
    policy_num_layers: int = 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or run 6-DoF Crazyflie PufferLib tuning sweeps")
    parser.add_argument("--pufferlib-root", default="../PufferLib-4-flightrl")
    parser.add_argument("--env-name", default="flightrl_sixdof_sweep")
    parser.add_argument("--build-mode", default="cpu", choices=("default", "float", "cpu"))
    parser.add_argument("--total-timesteps", type=int, default=2_097_152)
    parser.add_argument("--output", default="artifacts/replay/sixdof_puffer_sweep.json")
    parser.add_argument("--run", action="store_true", help="Execute commands instead of only writing the manifest")
    parser.add_argument("--no-build", action="store_true", help="Skip Puffer build for every executed variant")
    parser.add_argument("--max-variants", type=int, default=None)
    args = parser.parse_args()

    variants = default_variants()
    if args.max_variants is not None:
        variants = variants[: args.max_variants]
    records = []
    for index, variant in enumerate(variants):
        command = build_command(args, variant, skip_build=args.no_build or index > 0)
        record = {"variant": asdict(variant), "command": command}
        if args.run:
            record.update(run_command(command))
        records.append(record)

    report = {"total_timesteps": args.total_timesteps, "run": args.run, "records": records}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(report) + "\n")
    print(f"summary={output}")
    print(f"markdown={output.with_suffix('.md')}")


def default_variants() -> list[SweepVariant]:
    return [
        SweepVariant("small_h16_rr2_h64", 1024, 1, 1, 16, 2048, 2, 3e-4, 1e-3, 64),
        SweepVariant("base_h32_rr2_h128", 4096, 8, 8, 32, 8192, 2, 3e-4, 1e-3, 128),
        SweepVariant("fast_h32_rr1_h128", 4096, 8, 8, 32, 16384, 1, 7e-4, 3e-3, 128),
        SweepVariant("wide_h32_rr1_h256", 4096, 8, 8, 32, 16384, 1, 5e-4, 2e-3, 256),
        SweepVariant("large_h32_rr1_h128", 8192, 8, 8, 32, 16384, 1, 3e-4, 1e-3, 128),
    ]


def build_command(args: argparse.Namespace, variant: SweepVariant, *, skip_build: bool) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_sixdof_puffer4.py"),
        "--pufferlib-root",
        args.pufferlib_root,
        "--env-name",
        args.env_name,
        "--total-agents",
        str(variant.total_agents),
        "--num-buffers",
        str(variant.num_buffers),
        "--num-threads",
        str(variant.num_threads),
        "--policy-hidden-size",
        str(variant.policy_hidden_size),
        "--policy-num-layers",
        str(variant.policy_num_layers),
        "--build-mode",
        args.build_mode,
    ]
    if skip_build:
        command.append("--no-build")
    command.extend(
        [
            "--",
            "--train.total-timesteps",
            str(args.total_timesteps),
            "--train.horizon",
            str(variant.horizon),
            "--train.minibatch-size",
            str(variant.minibatch_size),
            "--train.replay-ratio",
            str(variant.replay_ratio),
            "--train.learning-rate",
            str(variant.learning_rate),
            "--train.ent-coef",
            str(variant.ent_coef),
        ]
    )
    return command


def run_command(command: list[str]) -> dict:
    start = perf_counter()
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    elapsed_s = perf_counter() - start
    combined = completed.stdout + "\n" + completed.stderr
    samples = parse_sps_samples(combined)
    train_samples = parse_train_sps_samples(combined)
    return {
        "returncode": completed.returncode,
        "elapsed_s": elapsed_s,
        "sps_samples": samples,
        "max_sps": max(samples) if samples else None,
        "train_sps_samples": train_samples,
        "max_train_sps": max(train_samples) if train_samples else None,
        "stdout_tail": tail_lines(completed.stdout),
        "stderr_tail": tail_lines(completed.stderr),
    }


def parse_sps_samples(text: str) -> list[float]:
    clean = ANSI_RE.sub("", text)
    return [float(value) * suffix_scale(suffix) for value, suffix in SPS_RE.findall(clean)]


def parse_train_sps_samples(text: str) -> list[float]:
    samples: list[float] = []
    for chunk in text.split("\x1b[0;0H"):
        clean = ANSI_RE.sub("", chunk)
        if not re.search(r"\bTrain\s+(?!0ms\b)[0-9]+(?:ms|s)\b", clean):
            continue
        match = SPS_RE.search(clean)
        if match:
            samples.append(float(match.group(1)) * suffix_scale(match.group(2)))
    return samples


def suffix_scale(suffix: str) -> float:
    return {"": 1.0, "K": 1_000.0, "M": 1_000_000.0, "G": 1_000_000_000.0}[suffix]


def tail_lines(text: str, limit: int = 20) -> list[str]:
    return text.splitlines()[-limit:]


def render_markdown(report: dict) -> str:
    lines = [
        "# 6-DoF Puffer Sweep",
        "",
        "| variant | agents | buffers | threads | horizon | minibatch | replay | lr | entropy | hidden | train SPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in report["records"]:
        variant = record["variant"]
        max_sps = record.get("max_train_sps")
        sps_text = f"{max_sps:.0f}" if max_sps else status_text(record)
        lines.append(
            f"| {variant['name']} | {variant['total_agents']} | {variant['num_buffers']} | {variant['num_threads']} | "
            f"{variant['horizon']} | {variant['minibatch_size']} | {variant['replay_ratio']} | "
            f"{variant['learning_rate']:.6g} | {variant['ent_coef']:.6g} | {variant['policy_hidden_size']} | "
            f"{sps_text} |"
        )
    lines.extend(["", "Commands are stored in the JSON report. Use `--run` to execute them."])
    return "\n".join(lines)


def status_text(record: dict) -> str:
    if "returncode" not in record:
        return "pending"
    if record["returncode"] == 0:
        return "no train SPS parsed"
    return f"failed rc={record['returncode']}"


if __name__ == "__main__":
    main()
