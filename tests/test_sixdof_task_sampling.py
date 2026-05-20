from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from flightrl.sixdof import SixDofPolicy
from flightrl.sixdof.dagger import collect_policy_dataset
from flightrl.sixdof.dataset import (
    collect_teacher_dataset,
    load_dataset,
    parse_task_probabilities,
    sample_task_indices,
    task_probability_vector,
)


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("position_yaw", "obstacle_avoidance", "circle")


def test_task_probability_vector_defaults_unspecified_tasks() -> None:
    probabilities = task_probability_vector(TASKS, {"position_yaw": 3.0})

    assert np.allclose(probabilities, [0.6, 0.2, 0.2])


def test_sample_task_indices_uses_probabilities() -> None:
    rng = np.random.default_rng(17)
    indices = sample_task_indices(rng, 1000, TASKS, task_probability_vector(TASKS, {"obstacle_avoidance": 5.0}))
    counts = np.bincount(indices, minlength=len(TASKS))

    assert counts[1] > counts[0]
    assert counts[1] > counts[2]


def test_parse_task_probabilities_rejects_invalid_weight() -> None:
    try:
        parse_task_probabilities(["position_yaw=0"])
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected invalid probability weight to fail")


def test_teacher_dataset_records_weighted_task_sampling() -> None:
    dataset = collect_teacher_dataset(
        task_spec="position_yaw,obstacle_avoidance,circle",
        num_envs=32,
        steps=4,
        seed=21,
        use_native_step=False,
        task_probabilities={"circle": 6.0},
    )
    counts = np.bincount(dataset["task_indices"], minlength=3)

    assert dataset["metadata"]["task_probability_weights"] == {"circle": 6.0}
    assert np.isclose(dataset["metadata"]["task_sampling_probabilities"]["circle"], 0.75)
    assert counts[2] > counts[0]
    assert counts[2] > counts[1]


def test_policy_dataset_records_weighted_task_sampling(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    torch.save(
        {
            "state_dict": SixDofPolicy(hidden_size=16, input_dim=31).state_dict(),
            "hidden_size": 16,
            "observation_dim": 31,
            "task": ",".join(TASKS),
            "tasks": list(TASKS),
        },
        checkpoint,
    )
    dataset = collect_policy_dataset(
        checkpoint_path=checkpoint,
        task_spec=None,
        num_envs=32,
        steps=4,
        seed=23,
        use_native_step=False,
        task_probabilities={"position_yaw": 6.0},
    )
    counts = np.bincount(dataset["task_indices"], minlength=3)

    assert dataset["metadata"]["task_probability_weights"] == {"position_yaw": 6.0}
    assert np.isclose(dataset["metadata"]["task_sampling_probabilities"]["position_yaw"], 0.75)
    assert counts[0] > counts[1]
    assert counts[0] > counts[2]


def test_teacher_dataset_cli_accepts_task_probability(tmp_path: Path) -> None:
    output = tmp_path / "weighted.npz"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_sixdof_teacher_dataset.py"),
            "--task",
            "position_yaw,obstacle_avoidance,circle",
            "--num-envs",
            "8",
            "--steps",
            "2",
            "--task-probability",
            "position_yaw=3.0",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    dataset = load_dataset(output)

    assert dataset["metadata"]["task_probability_weights"] == {"position_yaw": 3.0}
