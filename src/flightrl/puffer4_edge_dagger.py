from __future__ import annotations

import hashlib

import numpy as np
import torch

from flightrl.evidence_scope import require_existing_file_identity


def fixed_student_mask(metadata: dict, agents: int) -> np.ndarray:
    mix = metadata["execution_mix"]
    if (
        metadata["execution_policy"] != "dagger_student"
        or mix.get("schedule") != "fixed_per_agent_sha256_rank_v1"
    ):
        return np.zeros(agents, dtype=np.uint8)
    count = round(float(mix["student"]) * agents)
    ranked = sorted(
        range(agents),
        key=lambda index: hashlib.sha256(
            f"{mix['seed']}:{index}".encode()
        ).digest(),
    )
    mask = np.zeros(agents, dtype=np.uint8)
    mask[ranked[:count]] = 1
    return mask


@torch.no_grad()
def require_edge_execution_trace(dataset) -> None:
    agents = dataset.shape[1]
    expected_mask = fixed_student_mask(dataset.metadata, agents)
    if not np.array_equal(dataset.execution_student_mask, expected_mask):
        raise ValueError("edge dataset execution student mask does not reproduce")
    if dataset.metadata["execution_policy"] == "privileged_teacher":
        if not np.array_equal(dataset.behavior_actions, dataset.teacher_actions):
            raise ValueError("edge teacher behavior actions do not match labels")
        return
    actor = _load_execution_actor(dataset.metadata)
    state = actor.initial_state(agents)
    mask = torch.from_numpy(expected_mask.astype(bool, copy=False))
    for step in range(dataset.shape[0]):
        reset = torch.from_numpy(dataset.resets[step]).to(torch.bool).unsqueeze(1)
        state = torch.where(reset, torch.zeros_like(state), state)
        proposal, _grounding, state = actor.forward_step(
            dataset.model_observation(step),
            state,
        )
        expected = torch.from_numpy(dataset.teacher_actions[step]).clone()
        expected[mask] = proposal[mask]
        if not torch.equal(
            torch.from_numpy(dataset.behavior_actions[step]),
            expected,
        ):
            raise ValueError("edge DAgger behavior actions do not reproduce")


def _load_execution_actor(metadata: dict):
    try:
        identity = require_existing_file_identity(
            metadata["execution_checkpoint_identity"],
            label="edge DAgger execution checkpoint",
        )
    except OSError as exc:
        raise ValueError("edge DAgger execution checkpoint is unavailable") from exc
    from flightrl.puffer4_edge_checkpoint import load_edge_checkpoint
    from flightrl.puffer4_edge_native_build import (
        require_matching_edge_native_build_fingerprints,
    )

    actor, checkpoint = load_edge_checkpoint(identity["path"])
    require_existing_file_identity(
        identity,
        label="edge DAgger execution checkpoint",
    )
    if checkpoint.trained_target_ids != (0,):
        raise ValueError("edge DAgger execution checkpoint is not door-only")
    require_matching_edge_native_build_fingerprints(
        checkpoint.native_build_fingerprint,
        metadata["native_build_fingerprint"],
    )
    return actor
