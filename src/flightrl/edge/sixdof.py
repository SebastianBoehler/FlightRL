from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from flightrl.sixdof import SixDofPolicy


@dataclass(slots=True)
class EdgeExportResult:
    model_path: Path
    report_path: Path
    max_abs_error: float
    mean_abs_error: float


def export_sixdof_torchscript(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    report_path: str | Path | None = None,
    seed: int = 123,
    samples: int = 64,
) -> EdgeExportResult:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = _load_policy(checkpoint)
    example = _sample_observations(seed=seed, samples=samples)

    with torch.no_grad():
        expected = model.net(example)
        traced = torch.jit.trace(model.net, example)
        actual = traced(example)

    max_abs_error = float(torch.max(torch.abs(expected - actual)).item())
    mean_abs_error = float(torch.mean(torch.abs(expected - actual)).item())
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output))

    report = Path(report_path) if report_path else output.with_suffix(".parity.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "model": str(output),
                "task": checkpoint.get("task", "unknown"),
                "format": "torchscript-trace",
                "observation": {
                    "shape": [28],
                    "dtype": "float32",
                    "source": "6-DoF sim/state/ranger observation contract",
                },
                "action": {
                    "shape": [4],
                    "dtype": "float32",
                    "bounds": [-1.0, 1.0],
                    "meaning": ["thrust", "roll_rate", "pitch_rate", "yaw_rate"],
                },
                "parity": {
                    "samples": samples,
                    "max_abs_error": max_abs_error,
                    "mean_abs_error": mean_abs_error,
                },
                "safety": "Simulation export only; not approved for direct hardware control.",
            },
            indent=2,
        )
        + "\n"
    )
    return EdgeExportResult(output, report, max_abs_error, mean_abs_error)


def _load_policy(checkpoint: dict) -> SixDofPolicy:
    model = SixDofPolicy(hidden_size=int(checkpoint.get("hidden_size", 128)))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _sample_observations(*, seed: int, samples: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    observations = torch.empty((samples, 28), dtype=torch.float32).uniform_(-1.0, 1.0, generator=generator)
    observations[:, 6:10] = torch.nn.functional.normalize(observations[:, 6:10], dim=1)
    observations[:, 18:24] = torch.rand((samples, 6), generator=generator)
    return observations
