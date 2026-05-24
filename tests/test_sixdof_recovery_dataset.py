from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from flightrl.sixdof.dataset import collect_teacher_dataset, execution_actions, load_dataset


ROOT = Path(__file__).resolve().parents[1]


def test_execution_actions_adds_clipped_noise_without_mutating_labels() -> None:
    labels = np.zeros((8, 4), dtype=np.float32)
    rng = np.random.default_rng(3)
    executed = execution_actions(labels, rng, noise_std=5.0)

    assert executed.shape == labels.shape
    assert np.all(labels == 0.0)
    assert np.max(executed) <= 1.0
    assert np.min(executed) >= -1.0
    assert np.any(executed != labels)


def test_recovery_dataset_records_noisy_execution_metadata() -> None:
    dataset = collect_teacher_dataset(
        task_spec="position_yaw",
        num_envs=4,
        steps=3,
        seed=31,
        use_native_step=False,
        reset_profile="position_yaw_easy",
        observation_mode="history1",
        execution_noise_std=0.05,
    )

    assert dataset["observations"].shape[0] == 12
    assert dataset["actions"].shape == (12, 4)
    assert dataset["metadata"]["execution_policy"] == "noisy_teacher"
    assert dataset["metadata"]["execution_noise_std"] == 0.05


def test_recovery_profile_produces_explicit_recovery_dataset() -> None:
    dataset = collect_teacher_dataset(
        task_spec="position_yaw",
        num_envs=4,
        steps=3,
        seed=41,
        use_native_step=False,
        reset_profile="position_yaw_recovery",
        observation_mode="history1",
        execution_noise_std=0.08,
    )

    assert dataset["metadata"]["reset_profile"] == "position_yaw_recovery"
    assert dataset["metadata"]["observation_mode"] == "history1"
    assert dataset["metadata"]["terminal_fraction"] >= 0.0


def test_recovery_dataset_rejects_negative_noise() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        collect_teacher_dataset(
            task_spec="position_yaw",
            num_envs=2,
            steps=1,
            seed=7,
            use_native_step=False,
            execution_noise_std=-0.1,
        )


def test_teacher_dataset_cli_forwards_execution_noise(tmp_path: Path) -> None:
    output = tmp_path / "teacher_recovery.npz"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_sixdof_teacher_dataset.py"),
            "--task",
            "position_yaw",
            "--num-envs",
            "3",
            "--steps",
            "2",
            "--seed",
            "37",
            "--execution-noise-std",
            "0.03",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    metadata = json.loads(completed.stdout.strip().splitlines()[-1])
    loaded = load_dataset(output)
    assert metadata["execution_policy"] == "noisy_teacher"
    assert loaded["metadata"]["execution_noise_std"] == 0.03
