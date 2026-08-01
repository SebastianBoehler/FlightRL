from __future__ import annotations

from dataclasses import replace

import torch

from flightrl.evidence_scope import file_identity
from flightrl.puffer4_edge_checkpoint import load_edge_checkpoint
from flightrl.puffer4_edge_collection_evidence import (
    canonical_edge_environment_config,
)
from flightrl.puffer4_edge_dagger import fixed_student_mask
from flightrl.puffer4_edge_sequence import (
    edge_dataset_metadata,
    load_edge_sequence_dataset,
    require_edge_sequence_dataset,
)
from puffer4_edge_artifact_support import checkpoint_artifacts


def test_dagger_trace_reproduces_a_strict_bound_checkpoint(tmp_path) -> None:
    artifacts = checkpoint_artifacts(tmp_path)
    base = load_edge_sequence_dataset(artifacts.train)
    actor, _checkpoint = load_edge_checkpoint(artifacts.checkpoint)
    profile = base.metadata["collection_profile"]
    metadata = edge_dataset_metadata(
        split="train",
        base_seed=base.metadata["base_seed"],
        appearance_seed=base.metadata["appearance_seed"],
        steps=base.shape[0],
        agents=base.shape[1],
        target_ids=(0,),
        environment=base.metadata["environment"],
        native_build_fingerprint=base.metadata["native_build_fingerprint"],
        collection_profile=profile,
        environment_config=canonical_edge_environment_config(
            environment=base.metadata["environment"],
            agents=base.shape[1],
            base_seed=base.metadata["base_seed"],
            appearance_seed=base.metadata["appearance_seed"],
            collection_profile=profile,
        ),
        execution_policy="dagger_student",
        execution_checkpoint_identity=file_identity(artifacts.checkpoint),
        execution_student_fraction=0.5,
        execution_mix_seed=base.metadata["base_seed"],
    )
    mask = fixed_student_mask(metadata, base.shape[1])
    behavior = base.teacher_actions.copy()
    state = actor.initial_state(base.shape[1])
    selected_array = mask.astype(bool, copy=False)
    selected = torch.from_numpy(selected_array)
    with torch.no_grad():
        for step in range(base.shape[0]):
            reset = torch.from_numpy(base.resets[step]).to(torch.bool).unsqueeze(1)
            state = torch.where(reset, torch.zeros_like(state), state)
            proposal, _grounding, state = actor.forward_step(
                base.model_observation(step), state
            )
            behavior[step, selected_array] = proposal[selected].numpy()
    dataset = replace(
        base,
        behavior_actions=behavior,
        execution_student_mask=mask,
        metadata=metadata,
    )

    require_edge_sequence_dataset(dataset)
