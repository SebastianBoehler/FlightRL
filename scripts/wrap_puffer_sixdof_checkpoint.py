from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch

from flightrl.evidence_scope import file_identity
from flightrl.sixdof.checkpoint_contract import (
    PUFFER_POLICY_FORMAT,
    build_checkpoint_payload,
)
from flightrl.sixdof.env import ACTION_DIM, OBSERVATION_DIM
from flightrl.sixdof.observation import OBSERVATION_MODES, observation_dim
from flightrl.sixdof.puffer_policy import infer_metadata
from flightrl.sixdof.tasks import parse_task_spec, task_observation_dim


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wrap one raw PufferLib six-DoF state dict in the current desktop checkpoint contract"
    )
    parser.add_argument("--raw-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--observation-mode",
        choices=OBSERVATION_MODES,
        default="base",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw_checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output == raw_path:
        raise ValueError("wrapped output must not overwrite the raw Puffer checkpoint")
    if output.exists():
        raise FileExistsError(f"wrapped checkpoint already exists: {output}")
    raw = torch.load(raw_path, map_location="cpu")
    if not isinstance(raw, Mapping) or not raw:
        raise TypeError("raw Puffer checkpoint must be a non-empty state-dict mapping")
    if "state_dict" in raw:
        raise ValueError("input already has a checkpoint envelope; raw state dict required")
    state_dict = {
        str(key).removeprefix("module."): value
        for key, value in raw.items()
    }
    if not all(isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise TypeError("raw Puffer state dict values must all be tensors")

    tasks = parse_task_spec(args.task)
    network = infer_metadata(state_dict)
    expected_observation_dim = observation_dim(
        OBSERVATION_DIM + task_observation_dim(tasks),
        args.observation_mode,
    )
    if network.observation_dim != expected_observation_dim:
        raise ValueError(
            "raw Puffer observation dimension does not match the declared task and observation mode"
        )
    if network.action_dim != ACTION_DIM:
        raise ValueError("raw Puffer action dimension is not the current six-DoF action contract")

    checkpoint = build_checkpoint_payload(
        state_dict=state_dict,
        tasks=tasks,
        hidden_size=network.hidden_size,
        observation_mode=args.observation_mode,
        checkpoint_format=PUFFER_POLICY_FORMAT,
    )
    checkpoint.update(
        {
            "trainer": "pufferlib_external",
            "puffer_num_layers": network.num_layers,
            "source_raw_checkpoint": file_identity(raw_path),
            "note": "Imported desktop Puffer policy; no edge-v3 or live-hardware authority.",
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"wrapped_checkpoint={output}")


if __name__ == "__main__":
    main()
