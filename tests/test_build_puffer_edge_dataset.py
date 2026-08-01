from __future__ import annotations

import ctypes
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flightrl.puffer4_edge_dataset import EDGE_STUDENT_OBSERVATION_DIM
from flightrl.puffer4_edge_collection_evidence import (
    canonical_edge_environment_config,
)
from flightrl.puffer4_edge_policy import EdgeNavigationActor
from flightrl.puffer4_edge_schema import EDGE_FRAME_PIXELS, EDGE_OBSERVATION_DIM
from flightrl.puffer4_edge_sequence import (
    edge_dataset_metadata,
    require_edge_sequence_dataset,
)
from scripts.build_puffer_edge_dataset import (
    _DEFAULT_SEEDS,
    _execution_checkpoint,
    collect_dataset,
    main,
)
from puffer4_edge_artifact_support import native_build_fingerprint


def _native_observation(agents: int) -> torch.Tensor:
    values = torch.zeros(agents, EDGE_STUDENT_OBSERVATION_DIM)
    levels = torch.arange(EDGE_FRAME_PIXELS) % 16
    values[:, :EDGE_FRAME_PIXELS] = levels.to(torch.float32) / 15.0
    values[:, EDGE_FRAME_PIXELS + 8] = 1.0
    values[:, EDGE_FRAME_PIXELS + 14] = 1.0
    values[:, EDGE_FRAME_PIXELS + 19] = 1.0
    values[:, EDGE_OBSERVATION_DIM] = 0.8
    values[:, -1] = 64.0
    return values


class _FakeVec:
    obs_size = EDGE_STUDENT_OBSERVATION_DIM

    def __init__(self, observations: torch.Tensor, dones: list[list[int]]) -> None:
        self.observations = observations
        self.terminals = torch.zeros(observations.shape[0], dtype=torch.float32)
        self.obs_ptr = object()
        self.terminals_ptr = object()
        self.dones = dones
        self.executed: list[np.ndarray] = []
        self.closed = False

    def reset(self) -> None:
        self.terminals.zero_()

    def cpu_step(self, pointer: int) -> None:
        count = self.observations.shape[0] * 4
        buffer = (ctypes.c_float * count).from_address(pointer)
        self.executed.append(np.ctypeslib.as_array(buffer).reshape(-1, 4).copy())
        self.terminals[:] = torch.tensor(self.dones[len(self.executed) - 1])

    def close(self) -> None:
        self.closed = True


class _FakeTorchPuffer:
    def __init__(self, vec: _FakeVec) -> None:
        self.vec = vec
        self._C = SimpleNamespace(create_vec=lambda _args, _device: vec)
        self._C.gpu = False

    def _cpu_tensor(self, pointer, _shape, _dtype):
        return (
            self.vec.observations if pointer is self.vec.obs_ptr else self.vec.terminals
        )


def _metadata(
    *,
    steps: int,
    agents: int,
    dagger: bool = False,
    student_fraction: float = 1.0,
) -> dict:
    identity = {"path": "/tmp/student.pt", "sha256": "d" * 64} if dagger else None
    profile = {
        "obstacle_probability": 0.5,
        "camera_randomization": 1.0,
        "layout_diversity": 1.0,
    }
    environment = "edge-test"
    return edge_dataset_metadata(
        split="train",
        base_seed=11,
        appearance_seed=41,
        steps=steps,
        agents=agents,
        target_ids=(0,),
        environment=environment,
        native_build_fingerprint=native_build_fingerprint(
            Path("/tmp/flightrl-edge-collector-test"), environment
        ),
        collection_profile=profile,
        environment_config=canonical_edge_environment_config(
            environment=environment,
            agents=agents,
            base_seed=11,
            appearance_seed=41,
            collection_profile=profile,
        ),
        execution_policy="dagger_student" if dagger else "privileged_teacher",
        execution_checkpoint_identity=identity,
        execution_student_fraction=student_fraction if dagger else None,
        execution_mix_seed=11 if dagger else None,
    )


def _stateful_actor() -> EdgeNavigationActor:
    actor = EdgeNavigationActor(hidden_size=48)
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
        actor.fusion[0].bias.fill_(1.0)
        actor.recurrent.input_projection.bias[:48].fill_(1.0)
        actor.action_head[0].weight[0, 0] = 1.0
    actor.eval()
    return actor


def test_teacher_collection_rejects_native_contract_corruption() -> None:
    observations = _native_observation(1)
    observations[0, 0] = 0.5
    vec = _FakeVec(observations, [[0]])

    with pytest.raises(ValueError, match="gray4"):
        collect_dataset(
            {},
            _FakeTorchPuffer(vec),
            steps=1,
            agents=1,
            metadata=_metadata(steps=1, agents=1),
        )

    assert vec.closed is True
    assert vec.executed == []


def test_teacher_collection_derives_and_rejects_unsupported_native_target() -> None:
    observations = _native_observation(1)
    observations[0, EDGE_FRAME_PIXELS + 19 : EDGE_OBSERVATION_DIM] = torch.tensor(
        (0.0, 1.0, 0.0)
    )
    vec = _FakeVec(observations, [[0]])

    with pytest.raises(ValueError, match="target"):
        collect_dataset(
            {},
            _FakeTorchPuffer(vec),
            steps=1,
            agents=1,
            metadata=_metadata(steps=1, agents=1),
        )

    assert vec.executed == []


def test_teacher_collection_rejects_noncanonical_native_terminal() -> None:
    vec = _FakeVec(_native_observation(1), [[0.5]])

    with pytest.raises(ValueError, match="terminal"):
        collect_dataset(
            {},
            _FakeTorchPuffer(vec),
            steps=1,
            agents=1,
            metadata=_metadata(steps=1, agents=1),
        )


def test_dagger_collection_executes_fixed_mix_and_keeps_teacher_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flightrl.puffer4_edge_dagger._load_execution_actor",
        lambda _metadata: _stateful_actor(),
    )
    observations = _native_observation(4)
    vec = _FakeVec(observations, [[1] * 4, [0] * 4, [0] * 4])

    dataset = collect_dataset(
        {},
        _FakeTorchPuffer(vec),
        steps=3,
        agents=4,
        metadata=_metadata(
            steps=3,
            agents=4,
            dagger=True,
            student_fraction=0.5,
        ),
        execution_actor=_stateful_actor(),
    )

    np.testing.assert_allclose(dataset.teacher_actions[..., 0], 0.8)
    student_rows = vec.executed[0][:, 0] < 0.7
    assert int(student_rows.sum()) == 2
    assert vec.executed[0][student_rows, 0].tolist() == pytest.approx([0.5] * 2)
    assert vec.executed[1][student_rows, 0].tolist() == pytest.approx([0.5] * 2)
    assert vec.executed[2][student_rows, 0].tolist() == pytest.approx([0.75] * 2)
    for actions in vec.executed:
        assert actions[~student_rows, 0].tolist() == pytest.approx([0.8] * 2)
    np.testing.assert_array_equal(dataset.behavior_actions, np.asarray(vec.executed))
    np.testing.assert_array_equal(dataset.execution_student_mask, student_rows)
    assert dataset.resets[:, 0].tolist() == [1, 1, 0]
    assert dataset.dones[:, 0].tolist() == [1, 0, 0]
    assert dataset.metadata["execution_policy"] == "dagger_student"
    assert dataset.metadata["execution_mix"] == {
        "teacher": 0.5,
        "student": 0.5,
        "schedule": "fixed_per_agent_sha256_rank_v1",
        "seed": 11,
    }
    dataset.behavior_actions[0, student_rows, 0] += 0.01
    with pytest.raises(ValueError, match="behavior actions do not reproduce"):
        require_edge_sequence_dataset(dataset)


def test_checkpoint_loader_rejects_concurrent_file_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    path = tmp_path / "student.pt"
    identities = iter(
        (
            {"path": str(path), "sha256": "a" * 64},
            {"path": str(path), "sha256": "b" * 64},
        )
    )
    monkeypatch.setattr(
        "scripts.build_puffer_edge_dataset.file_identity",
        lambda _path: next(identities),
    )
    monkeypatch.setattr(
        "scripts.build_puffer_edge_dataset.load_edge_checkpoint",
        lambda _path: (_stateful_actor(), SimpleNamespace(trained_target_ids=(0,))),
    )

    with pytest.raises(RuntimeError, match="changed while loading"):
        _execution_checkpoint(path)


def test_dagger_collection_rejects_output_alias_before_loading_checkpoint(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "student.pt"
    original = b"bound checkpoint"
    checkpoint.write_bytes(original)

    with pytest.raises(ValueError, match="artifact paths must be distinct"):
        main(
            [
                "--split",
                "train",
                "--execution-checkpoint",
                str(checkpoint),
                "--output",
                str(checkpoint),
            ]
        )

    assert checkpoint.read_bytes() == original


def test_collector_rejects_execution_mode_metadata_mismatch() -> None:
    vec = _FakeVec(_native_observation(1), [[0]])

    with pytest.raises(ValueError, match="execution actor"):
        collect_dataset(
            {},
            _FakeTorchPuffer(vec),
            steps=1,
            agents=1,
            metadata=_metadata(steps=1, agents=1, dagger=True),
        )


def test_passive_replay_defaults_are_disjoint_from_closed_loop_profiles() -> None:
    from flightrl.puffer4_edge_evaluation_gate import EDGE_EVALUATION_PROFILES

    evaluation_physical = {record[1] for record in EDGE_EVALUATION_PROFILES}
    evaluation_appearance = {record[2] for record in EDGE_EVALUATION_PROFILES}

    assert _DEFAULT_SEEDS["final"][0] not in evaluation_physical
    assert _DEFAULT_SEEDS["final"][1] not in evaluation_appearance
