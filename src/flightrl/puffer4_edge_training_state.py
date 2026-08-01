from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import struct

import torch


EDGE_STATE_DIGEST_SCHEMA = "flightrl.edge_v3.actor_state_sha256.v1"


def edge_state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash sorted tensor names, dtypes, shapes, and exact contiguous bytes."""
    if not isinstance(state_dict, Mapping):
        raise TypeError("edge actor state_dict must be a mapping")
    digest = hashlib.sha256()
    digest.update(EDGE_STATE_DIGEST_SCHEMA.encode("ascii") + b"\0")
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(name, str) or not name:
            raise ValueError("edge actor state tensor names must be nonempty strings")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"edge actor state {name!r} must be a tensor")
        exact = tensor.detach().cpu().contiguous()
        header = json.dumps(
            [name, str(exact.dtype), list(exact.shape)],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        raw = exact.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        digest.update(struct.pack(">Q", len(header)))
        digest.update(header)
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()
