from __future__ import annotations

import argparse
import json
from pathlib import Path

from flightrl.replay import fit_linear_calibration, load_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit scale/bias calibration from aligned Crazyflie real/sim replay CSVs")
    parser.add_argument("--real", required=True)
    parser.add_argument("--sim", required=True)
    parser.add_argument("--output", default="artifacts/replay/replay_calibration.json")
    parser.add_argument("--signals", nargs="*", default=None)
    args = parser.parse_args()

    real_rows = load_rows(args.real)
    sim_rows = load_rows(args.sim)
    result = {
        "real": args.real,
        "sim": args.sim,
        "calibration": fit_linear_calibration(real_rows, sim_rows, args.signals),
        "model": "real ~= scale * sim + bias",
        "safety": "Replay calibration scaffold only; do not use as a hardware deployment gate without matched flights.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".md").write_text(render_markdown(result) + "\n")
    print(f"summary={output}")
    print(f"markdown={output.with_suffix('.md')}")


def render_markdown(result: dict) -> str:
    lines = [
        "# Replay Calibration",
        "",
        f"- Real: `{result['real']}`",
        f"- Sim: `{result['sim']}`",
        f"- Model: `{result['model']}`",
        "",
        "| signal | samples | scale | bias | raw RMSE | fitted RMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, fit in result["calibration"]["signals"].items():
        lines.append(
            f"| {key} | {fit['samples']} | {fit['scale']:.6g} | {fit['bias']:.6g} | "
            f"{fit['raw_rmse']:.6g} | {fit['fitted_rmse']:.6g} |"
        )
    lines.extend(["", result["safety"]])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
