from __future__ import annotations

import argparse
from pathlib import Path

from flightrl.edge import export_sixdof_torchscript


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a 6-DoF policy checkpoint to TorchScript with parity report")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    output = Path(args.output) if args.output else Path("artifacts/edge") / f"{checkpoint.stem}.ts"
    result = export_sixdof_torchscript(
        checkpoint,
        output,
        report_path=args.report,
        seed=args.seed,
        samples=args.samples,
    )
    print(f"model={result.model_path}")
    print(f"report={result.report_path}")
    print(f"max_abs_error={result.max_abs_error:.8f}")
    print(f"mean_abs_error={result.mean_abs_error:.8f}")


if __name__ == "__main__":
    main()
