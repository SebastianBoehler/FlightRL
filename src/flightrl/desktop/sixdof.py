from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from flightrl.evidence_scope import DESKTOP_CPU_SCOPE, file_identity
from flightrl.sixdof import load_policy_from_checkpoint


@dataclass(slots=True)
class DesktopExportResult:
    model_path: Path
    report_path: Path
    max_abs_error: float
    mean_abs_error: float


def export_sixdof_desktop_torchscript(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    report_path: str | Path | None = None,
    seed: int = 123,
    samples: int = 64,
) -> DesktopExportResult:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = load_policy_from_checkpoint(checkpoint)
    observation_dim = int(checkpoint.get("observation_dim", 28))
    observation_mode = str(checkpoint.get("observation_mode", "base"))
    controller = str(checkpoint.get("controller", "policy"))
    task_count = len(checkpoint.get("tasks", [])) if checkpoint.get("task_conditioned", False) else 1
    example = sample_sixdof_observations(seed=seed, samples=samples, observation_dim=observation_dim, observation_mode=observation_mode, task_count=task_count)

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
                "schema": "flightrl.sixdof.desktop_export.v1",
                "evidence_scope": DESKTOP_CPU_SCOPE,
                "deployment_authority": False,
                "checkpoint": file_identity(checkpoint_path),
                "model": file_identity(output),
                "task": checkpoint.get("task", "unknown"),
                "controller": controller,
                "residual_scale": checkpoint.get("residual_scale"),
                "format": "torchscript-trace",
                "observation": {
                    "shape": [observation_dim],
                    "dtype": "float32",
                    "source": "6-DoF sim/state/ranger observation contract",
                    "mode": observation_mode,
                    "task_conditioned": bool(checkpoint.get("task_conditioned", False)),
                },
                "action": {
                    "shape": [4],
                    "dtype": "float32",
                    "bounds": [-1.0, 1.0],
                    "meaning": action_meaning(controller),
                },
                "parity": {
                    "samples": samples,
                    "max_abs_error": max_abs_error,
                    "mean_abs_error": mean_abs_error,
                },
                "safety": safety_text(controller),
            },
            indent=2,
        )
        + "\n"
    )
    return DesktopExportResult(output, report, max_abs_error, mean_abs_error)


def action_meaning(controller: str) -> list[str]:
    if controller == "teacher_residual":
        return ["thrust_residual", "roll_rate_residual", "pitch_rate_residual", "yaw_rate_residual"]
    return ["thrust", "roll_rate", "pitch_rate", "yaw_rate"]


def safety_text(controller: str) -> str:
    if controller == "teacher_residual":
        return "Desktop CPU residual-actor export only; the analytic teacher is absent, and this is neither AI Deck deployment evidence nor live-hardware authority."
    return "Desktop CPU export only; this is neither AI Deck deployment evidence nor live-hardware authority."


def sample_sixdof_observations(*, seed: int, samples: int, observation_dim: int, observation_mode: str, task_count: int) -> torch.Tensor:
    base_dim = 28 + (task_count if task_count > 1 else 0)
    if observation_mode == "history1":
        current = _sample_base(seed=seed, samples=samples, observation_dim=base_dim, task_count=task_count)
        delta = torch.zeros_like(current)
        previous_action = torch.zeros((samples, 4), dtype=torch.float32)
        return torch.cat([current, delta, previous_action], dim=1)
    return _sample_base(seed=seed, samples=samples, observation_dim=observation_dim, task_count=task_count)


def _sample_base(*, seed: int, samples: int, observation_dim: int, task_count: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    observations = torch.empty((samples, observation_dim), dtype=torch.float32).uniform_(-1.0, 1.0, generator=generator)
    observations[:, 6:10] = torch.nn.functional.normalize(observations[:, 6:10], dim=1)
    observations[:, 18:24] = torch.rand((samples, 6), generator=generator)
    if task_count > 1:
        task_dim = task_count
        observations[:, 28:] = 0.0
        observations[torch.arange(samples), 28 + torch.arange(samples) % task_dim] = 1.0
    return observations
