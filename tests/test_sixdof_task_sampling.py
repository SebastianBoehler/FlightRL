from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from flightrl.sixdof import SixDofPolicy, build_checkpoint_payload
from flightrl.sixdof.dagger import collect_policy_dataset
from flightrl.sixdof.circle import circle_tangent_yaw_from_arrays
from flightrl.sixdof.dataset import (
    collect_teacher_dataset,
    load_dataset,
    parse_task_probabilities,
    task_probability_vector,
)
from flightrl.sixdof.env import quat_to_yaw, wrap_angle
from flightrl.sixdof.episode_tasks import EpisodeTaskAssignments, sample_task_indices
from flightrl.sixdof.tasks import TASKS as ACTIVE_TASKS, parse_task_spec


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("position_yaw", "obstacle_avoidance", "circle")


def test_multitask_contains_only_tasks_with_explicit_mdp_contracts() -> None:
    assert ACTIVE_TASKS == TASKS
    assert parse_task_spec("multitask") == TASKS
    with pytest.raises(ValueError, match="unknown 6-DoF task"):
        parse_task_spec("attitude")


def test_task_probability_vector_defaults_unspecified_tasks() -> None:
    probabilities = task_probability_vector(TASKS, {"position_yaw": 3.0})

    assert np.allclose(probabilities, [0.6, 0.2, 0.2])


def test_sample_task_indices_uses_probabilities() -> None:
    rng = np.random.default_rng(17)
    indices = sample_task_indices(rng, 1000, TASKS, task_probability_vector(TASKS, {"obstacle_avoidance": 5.0}))
    counts = np.bincount(indices, minlength=len(TASKS))

    assert counts[1] > counts[0]
    assert counts[1] > counts[2]


@pytest.mark.parametrize(
    "probabilities",
    (
        np.asarray([1.0, 1.0]),
        np.asarray([1.0, np.nan, 1.0]),
        np.asarray([1.0, -0.1, 1.0]),
    ),
)
def test_episode_task_assignments_reject_invalid_probabilities(
    probabilities: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="probabilities"):
        EpisodeTaskAssignments.sample(
            rng=np.random.default_rng(19),
            num_envs=8,
            tasks=TASKS,
            probabilities=probabilities,
        )


def test_episode_task_assignments_resample_only_finished_rows() -> None:
    assignments = EpisodeTaskAssignments.sample(
        rng=np.random.default_rng(19),
        num_envs=8,
        tasks=TASKS,
    )
    before = assignments.indices.copy()
    done = np.asarray([False, True, False, False, True, False, True, False])

    assignments.resample(done)

    assert np.array_equal(assignments.indices[~done], before[~done])


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


def test_teacher_dataset_keeps_task_id_fixed_within_each_episode() -> None:
    dataset = collect_teacher_dataset(
        task_spec="position_yaw,obstacle_avoidance,circle",
        num_envs=24,
        steps=5,
        seed=27,
        use_native_step=False,
    )

    by_step = dataset["task_indices"].reshape(5, 24)
    assert np.all(by_step == by_step[0])


def test_teacher_dataset_circle_observation_uses_tangent_yaw_context() -> None:
    dataset = collect_teacher_dataset(
        task_spec="position_yaw,circle",
        num_envs=64,
        steps=1,
        seed=29,
        use_native_step=False,
    )
    circle = dataset["task_indices"] == 1
    obs = dataset["observations"][circle]
    assert len(obs) > 0

    target = obs[:, 13:16] * np.asarray([2.0, 2.0, 2.5], dtype=np.float32)
    position = target - obs[:, :3] * np.asarray([2.0, 2.0, 1.5], dtype=np.float32)
    yaw_error = wrap_angle(
        circle_tangent_yaw_from_arrays(position, target) - quat_to_yaw(obs[:, 6:10])
    )
    assert np.allclose(obs[:, 16], np.sin(yaw_error), atol=1e-6)
    assert np.allclose(obs[:, 17], np.cos(yaw_error), atol=1e-6)


def test_policy_dataset_records_weighted_task_sampling(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    torch.save(
        build_checkpoint_payload(
            state_dict=SixDofPolicy(hidden_size=16, input_dim=31).state_dict(),
            tasks=TASKS,
            hidden_size=16,
        ),
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


def test_policy_dataset_keeps_task_id_fixed_within_each_episode(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    torch.save(
        build_checkpoint_payload(
            state_dict=SixDofPolicy(hidden_size=16, input_dim=31).state_dict(),
            tasks=TASKS,
            hidden_size=16,
        ),
        checkpoint,
    )
    dataset = collect_policy_dataset(
        checkpoint_path=checkpoint,
        task_spec=None,
        num_envs=24,
        steps=5,
        seed=31,
        use_native_step=False,
    )

    by_step = dataset["task_indices"].reshape(5, 24)
    assert np.all(by_step == by_step[0])


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
