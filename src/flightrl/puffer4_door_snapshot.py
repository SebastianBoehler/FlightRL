from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Mapping

import torch


@dataclass(frozen=True, slots=True)
class FixedDoorCheckpointSnapshot:
    source_path: Path
    sha256: str
    state_dict: Mapping[str, torch.Tensor]


def load_fixed_door_checkpoint_snapshot(
    checkpoint: str | Path,
    expected_sha256: str,
) -> FixedDoorCheckpointSnapshot:
    path = Path(checkpoint).resolve()
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            "fixed-door checkpoint snapshot SHA-256 does not match bundle"
        )
    try:
        state = torch.load(
            BytesIO(data),
            map_location="cpu",
            weights_only=True,
        )
    except (EOFError, RuntimeError, TypeError) as exc:
        raise ValueError(
            "fixed-door checkpoint snapshot is unreadable"
        ) from exc
    if not isinstance(state, Mapping):
        raise ValueError("fixed-door checkpoint snapshot has no state dict")
    return FixedDoorCheckpointSnapshot(path, digest, state)
