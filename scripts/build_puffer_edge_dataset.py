from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path

import numpy as np
import torch

from flightrl import puffer4_door_runner as door_runner
from flightrl.artifact_paths import require_distinct_artifact_paths
from flightrl.evidence_scope import file_identity
from flightrl.puffer4_config import Puffer4ExportSettings
from flightrl.puffer4_door_mission import FIXED_DOOR_MISSION_METRIC_V1
from flightrl.puffer4_edge_checkpoint import load_edge_checkpoint
from flightrl.puffer4_edge_dataset import (
    EDGE_STUDENT_OBSERVATION_DIM,
    adapt_native_door_observation_batch,
)
from flightrl.puffer4_edge_dagger import fixed_student_mask
from flightrl.puffer4_edge_policy import EdgeNavigationActor
from flightrl.puffer4_edge_native_build import require_current_edge_native_build_fingerprint
from flightrl.puffer4_edge_schema import EDGE_OBSERVATION_DIM
from flightrl.puffer4_edge_sequence import (
    EdgeSequenceDataset,
    edge_dataset_metadata,
    require_edge_sequence_dataset,
    write_edge_sequence_dataset,
)
from flightrl.puffer4_edge_student_export import (
    EDGE_STUDENT_NATIVE_FILES,
    export_edge_student_assets,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUFFER = ROOT.parent / "PufferLib-4-edge-v3"
_DEFAULT_SEEDS = {
    "train": (11_001, 41_001),
    "selection": (21_001, 51_001),
    "final": (34_001, 64_001),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Collect an exact edge-v3 recurrent door-teacher dataset"
    )
    parser.add_argument("--puffer-root", type=Path, default=DEFAULT_PUFFER)
    parser.add_argument("--env-name", default="flightrl_edge_v3_door_student")
    parser.add_argument("--split", choices=tuple(_DEFAULT_SEEDS), required=True)
    parser.add_argument("--agents", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2600)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--appearance-seed", type=int)
    parser.add_argument("--obstacle-probability", type=float, default=0.5)
    parser.add_argument("--camera-randomization", type=float, default=1.0)
    parser.add_argument("--layout-diversity", type=float, default=1.0)
    parser.add_argument("--execution-checkpoint", type=Path)
    parser.add_argument("--student-fraction", type=float)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    default_seed, default_appearance = _DEFAULT_SEEDS[args.split]
    seed = args.seed if args.seed is not None else default_seed
    appearance_seed = (
        args.appearance_seed if args.appearance_seed is not None else default_appearance
    )
    suffix = "_dagger" if args.execution_checkpoint is not None else ""
    output = args.output or Path(
        f"artifacts/edge_v3/edge_door_{args.split}{suffix}_s{seed}.npz"
    )
    if args.execution_checkpoint is not None:
        resolved = require_distinct_artifact_paths(
            execution_checkpoint=args.execution_checkpoint,
            output=output,
        )
        args.execution_checkpoint = resolved["execution_checkpoint"]
        output = resolved["output"]
    execution_actor, execution_identity = _execution_checkpoint(args.execution_checkpoint)
    student_fraction = args.student_fraction
    if execution_actor is not None and student_fraction is None:
        student_fraction = 0.75

    settings = Puffer4ExportSettings(
        env_name=args.env_name,
        total_agents=args.agents,
        num_buffers=1,
        num_threads=min(args.agents, 8),
        policy_hidden_size=48,
        train_seed=17,
    )
    export_edge_student_assets(args.puffer_root, settings)
    builder = door_runner.build_environment
    if args.skip_build:
        builder = door_runner.verify_native_build
    fingerprint = builder(
        args.puffer_root,
        args.env_name,
        EDGE_STUDENT_NATIVE_FILES,
    )
    puffer_args, torch_pufferl = door_runner.load_puffer(
        args.puffer_root,
        args.env_name,
        EDGE_STUDENT_NATIVE_FILES,
    )
    require_current_edge_native_build_fingerprint(fingerprint)
    collection_profile = {
        "obstacle_probability": args.obstacle_probability,
        "camera_randomization": args.camera_randomization,
        "layout_diversity": args.layout_diversity,
    }
    puffer_args["env"].update(
        {
            "seed": seed,
            "appearance_seed": appearance_seed,
            **collection_profile,
            "camera_mask": 0.0,
        }
    )
    puffer_args["vec"]["total_agents"] = args.agents
    FIXED_DOOR_MISSION_METRIC_V1.verify_env(puffer_args["env"])
    dataset = collect_dataset(
        puffer_args,
        torch_pufferl,
        steps=args.steps,
        agents=args.agents,
        metadata=edge_dataset_metadata(
            split=args.split,
            base_seed=seed,
            appearance_seed=appearance_seed,
            steps=args.steps,
            agents=args.agents,
            target_ids=(0,),
            environment=args.env_name,
            native_build_fingerprint=fingerprint,
            collection_profile=collection_profile,
            environment_config=puffer_args["env"],
            execution_policy=(
                "dagger_student" if execution_actor else "privileged_teacher"
            ),
            execution_checkpoint_identity=execution_identity,
            execution_student_fraction=student_fraction,
            execution_mix_seed=seed if execution_actor is not None else None,
        ),
        execution_actor=execution_actor,
    )
    require_current_edge_native_build_fingerprint(fingerprint)
    write_edge_sequence_dataset(output, dataset)
    print(json.dumps(dataset.metadata, indent=2, sort_keys=True))
    print(json.dumps(file_identity(output), sort_keys=True))


def collect_dataset(
    args: dict,
    torch_pufferl,
    *,
    steps: int,
    agents: int,
    metadata: dict,
    execution_actor: EdgeNavigationActor | None = None,
) -> EdgeSequenceDataset:
    dagger = metadata.get("execution_policy") == "dagger_student"
    if dagger != (execution_actor is not None):
        raise ValueError("edge dataset execution actor does not match metadata")
    if execution_actor is not None and type(execution_actor) is not EdgeNavigationActor:
        raise TypeError("edge dataset execution actor is incompatible")
    if execution_actor is not None:
        execution_actor.eval()
    vec = torch_pufferl._C.create_vec(args, torch_pufferl._C.gpu)
    try:
        if vec.obs_size != EDGE_STUDENT_OBSERVATION_DIM:
            raise RuntimeError("native edge student observation size is incompatible")
        observations = torch_pufferl._cpu_tensor(
            vec.obs_ptr,
            (agents, vec.obs_size),
            torch.float32,
        )
        terminals = torch_pufferl._cpu_tensor(
            vec.terminals_ptr,
            (agents,),
            torch.float32,
        )
        if terminals.shape != (agents,) or terminals.dtype != torch.float32:
            raise ValueError("native edge terminal buffer is incompatible")
        arrays = _allocate(steps, agents)
        reset = np.ones(agents, dtype=np.uint8)
        state = execution_actor.initial_state(agents) if execution_actor else None
        mask = fixed_student_mask(metadata, agents)
        arrays["execution_student_mask"][:] = mask
        student_mask = torch.from_numpy(mask.astype(bool)).unsqueeze(1)
        vec.reset()
        for step in range(steps):
            batch = adapt_native_door_observation_batch(observations)
            if np.any(~np.isin(batch.target_ids, metadata["target_ids"])):
                raise ValueError("native edge target is outside dataset target IDs")
            arrays["packed_frames"][step] = batch.packed_frames
            arrays["telemetry"][step] = batch.telemetry
            arrays["target_ids"][step] = batch.target_ids
            arrays["teacher_actions"][step] = batch.teacher_actions
            arrays["grounding"][step] = batch.grounding
            arrays["resets"][step] = reset
            teacher_actions = torch.from_numpy(batch.teacher_actions)
            if execution_actor is not None:
                student_actions, state = _student_actions(
                    execution_actor,
                    observations,
                    state,
                    reset,
                )
                actions = torch.where(
                    student_mask,
                    student_actions,
                    teacher_actions,
                ).contiguous()
            else:
                actions = teacher_actions.contiguous()
            arrays["behavior_actions"][step] = actions.detach().numpy()
            vec.cpu_step(actions.data_ptr())
            terminal_values = terminals.detach().numpy()
            invalid_terminal = (terminal_values != 0.0) & (terminal_values != 1.0)
            if not np.isfinite(terminal_values).all() or np.any(invalid_terminal):
                raise ValueError(
                    "native edge terminal flags must be finite binary values"
                )
            done = terminal_values.astype(np.uint8, copy=True)
            arrays["dones"][step] = done
            reset = done
    finally:
        vec.close()
    dataset = EdgeSequenceDataset(metadata=metadata, **arrays)
    require_edge_sequence_dataset(dataset)
    return dataset


def _allocate(steps: int, agents: int) -> dict[str, np.ndarray]:
    prefix = (steps, agents)
    return {
        "packed_frames": np.empty(prefix + (1536,), dtype=np.uint8),
        "telemetry": np.empty(prefix + (19,), dtype=np.float32),
        "target_ids": np.empty(prefix, dtype=np.uint8),
        "teacher_actions": np.empty(prefix + (4,), dtype=np.float32),
        "behavior_actions": np.empty(prefix + (4,), dtype=np.float32),
        "execution_student_mask": np.empty(agents, dtype=np.uint8),
        "grounding": np.empty(prefix + (4,), dtype=np.float32),
        "resets": np.empty(prefix, dtype=np.uint8),
        "dones": np.empty(prefix, dtype=np.uint8),
    }


@torch.no_grad()
def _student_actions(
    actor: EdgeNavigationActor,
    observation: torch.Tensor,
    state: torch.Tensor,
    reset: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    flags = torch.from_numpy(reset).to(torch.bool).unsqueeze(1)
    state = torch.where(flags, torch.zeros_like(state), state)
    actions, _grounding, state = actor.forward_step(
        observation[:, :EDGE_OBSERVATION_DIM].clone(),
        state,
    )
    return actions.contiguous(), state


def _execution_checkpoint(
    path: Path | None,
) -> tuple[EdgeNavigationActor | None, dict[str, str] | None]:
    if path is None:
        return None, None
    identity = file_identity(path)
    actor, checkpoint = load_edge_checkpoint(path)
    if file_identity(path) != identity:
        raise RuntimeError("edge DAgger checkpoint changed while loading")
    if checkpoint.trained_target_ids != (0,):
        raise ValueError("edge DAgger checkpoint must be explicitly door-only")
    return actor, identity


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.agents <= 0 or args.steps <= 0:
        parser.error("agents and steps must be positive")
    for name in ("seed", "appearance_seed"):
        value = getattr(args, name)
        if value is not None and not 0 <= value < 2**32:
            parser.error(f"{name.replace('_', '-')} must be uint32")
    for name in ("obstacle_probability", "camera_randomization", "layout_diversity"):
        value = getattr(args, name)
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            parser.error(f"{name.replace('_', '-')} must be finite in [0, 1]")
    if args.execution_checkpoint is not None and args.split != "train":
        parser.error("--execution-checkpoint is restricted to --split train")
    if args.execution_checkpoint is None and args.student_fraction is not None:
        parser.error("--student-fraction requires --execution-checkpoint")


if __name__ == "__main__":
    main()
