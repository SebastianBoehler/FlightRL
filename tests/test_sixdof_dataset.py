from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from flightrl.sixdof import SixDofPolicy
from flightrl.sixdof.dagger import collect_policy_dataset
from flightrl.sixdof.dataset import collect_teacher_dataset, load_dataset, write_dataset
from flightrl.sixdof.offline import OfflineTrainConfig, checkpoint_score


ROOT = Path(__file__).resolve().parents[1]


def test_collect_teacher_dataset_roundtrip(tmp_path: Path) -> None:
    dataset = collect_teacher_dataset(
        task_spec="position_yaw,obstacle_avoidance",
        num_envs=4,
        steps=3,
        seed=5,
        use_native_step=False,
    )
    path = write_dataset(tmp_path / "teacher.npz", dataset)
    loaded = load_dataset(path)

    assert loaded["observations"].shape == (12, 30)
    assert loaded["actions"].shape == (12, 4)
    assert loaded["metadata"]["tasks"] == ["position_yaw", "obstacle_avoidance"]
    assert np.array_equal(loaded["task_indices"], dataset["task_indices"])


def test_action_gap_cli_reports_per_task(tmp_path: Path) -> None:
    dataset = collect_teacher_dataset(task_spec="position_yaw", num_envs=4, steps=2, seed=7, use_native_step=False)
    dataset_path = write_dataset(tmp_path / "teacher.npz", dataset)
    checkpoint = tmp_path / "policy.pt"
    torch.save(
        {
            "state_dict": SixDofPolicy(hidden_size=16).state_dict(),
            "hidden_size": 16,
            "observation_dim": 28,
            "task": "position_yaw",
            "tasks": ["position_yaw"],
        },
        checkpoint,
    )
    report_path = tmp_path / "gap.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_sixdof_action_gap.py"),
            "--checkpoint",
            str(checkpoint),
            "--dataset",
            str(dataset_path),
            "--output",
            str(report_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(report_path.read_text())
    assert report["samples"] == 8
    assert "position_yaw" in report["per_task"]


def test_offline_training_cli_writes_checkpoint(tmp_path: Path) -> None:
    dataset = collect_teacher_dataset(task_spec="position_yaw", num_envs=4, steps=3, seed=9, use_native_step=False)
    dataset_path = write_dataset(tmp_path / "teacher.npz", dataset)
    checkpoint = tmp_path / "offline.pt"
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_sixdof_offline.py"),
            "--dataset",
            str(dataset_path),
            "--checkpoint",
            str(checkpoint),
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--hidden-size",
            "16",
            "--eval-steps",
            "4",
            "--eval-num-envs",
            "4",
            "--select-by-eval",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    saved = torch.load(checkpoint, map_location="cpu")
    assert "checkpoint=" in run.stdout
    assert saved["dataset"] == str(dataset_path)
    assert saved["val_loss"] >= 0.0
    assert saved["selection_mode"] == "eval"
    assert saved["selection_metrics"] is not None


def test_eval_selected_checkpoint_score_prioritizes_survival_and_clearance() -> None:
    config = OfflineTrainConfig(dataset="dummy", select_by_eval=True)
    safer = checkpoint_payload(completed=0.9, clearance=0.2, position_error=2.0)
    precise_crash = checkpoint_payload(completed=0.2, clearance=0.01, position_error=0.2)
    assert checkpoint_score(safer, config) < checkpoint_score(precise_crash, config)


def test_collect_policy_dataset_roundtrip(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    torch.save(
        {
            "state_dict": SixDofPolicy(hidden_size=16).state_dict(),
            "hidden_size": 16,
            "observation_dim": 28,
            "task": "position_yaw",
            "tasks": ["position_yaw"],
        },
        checkpoint,
    )
    dataset = collect_policy_dataset(
        checkpoint_path=checkpoint,
        task_spec=None,
        num_envs=4,
        steps=3,
        seed=11,
        use_native_step=False,
    )
    path = write_dataset(tmp_path / "dagger.npz", dataset)
    loaded = load_dataset(path)

    assert loaded["observations"].shape == (12, 28)
    assert loaded["actions"].shape == (12, 4)
    assert loaded["metadata"]["rollout_policy"] == "checkpoint"
    assert loaded["metadata"]["source_checkpoint"] == str(checkpoint)


def checkpoint_payload(*, completed: float, clearance: float, position_error: float) -> dict:
    return {
        "val_loss": 0.1,
        "selection_metrics": {
            "mean_completed_fraction": completed,
            "clearance_p01_m": clearance,
            "min_clearance_m": clearance,
            "mean_position_error_m": position_error,
        },
    }


def test_dagger_dataset_cli_appends_compatible_dataset(tmp_path: Path) -> None:
    base = collect_teacher_dataset(task_spec="position_yaw", num_envs=4, steps=2, seed=13, use_native_step=False)
    base_path = write_dataset(tmp_path / "base.npz", base)
    checkpoint = tmp_path / "policy.pt"
    torch.save(
        {
            "state_dict": SixDofPolicy(hidden_size=16).state_dict(),
            "hidden_size": 16,
            "observation_dim": 28,
            "task": "position_yaw",
            "tasks": ["position_yaw"],
        },
        checkpoint,
    )
    output = tmp_path / "merged.npz"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_sixdof_dagger_dataset.py"),
            "--checkpoint",
            str(checkpoint),
            "--append-dataset",
            str(base_path),
            "--output",
            str(output),
            "--num-envs",
            "4",
            "--steps",
            "2",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    merged = load_dataset(output)
    assert merged["observations"].shape == (16, 28)
    assert merged["metadata"]["samples"] == 16


def test_dagger_training_cli_writes_iteration_report(tmp_path: Path) -> None:
    dataset = collect_teacher_dataset(task_spec="position_yaw", num_envs=4, steps=3, seed=15, use_native_step=False)
    dataset_path = write_dataset(tmp_path / "seed.npz", dataset)
    initial = tmp_path / "initial.pt"
    torch.save(
        {
            "state_dict": SixDofPolicy(hidden_size=16).state_dict(),
            "hidden_size": 16,
            "observation_dim": 28,
            "task": "position_yaw",
            "tasks": ["position_yaw"],
        },
        initial,
    )
    output_dir = tmp_path / "dagger"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_sixdof_dagger.py"),
            "--seed-dataset",
            str(dataset_path),
            "--initial-checkpoint",
            str(initial),
            "--output-dir",
            str(output_dir),
            "--iterations",
            "1",
            "--num-envs",
            "4",
            "--steps",
            "2",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--hidden-size",
            "16",
            "--eval-steps",
            "4",
            "--eval-num-envs",
            "4",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((output_dir / "summary.json").read_text())
    assert (output_dir / "iter_01.pt").exists()
    assert summary["iterations"][0]["iteration"] == 1
    assert "gate" in summary["iterations"][0]
