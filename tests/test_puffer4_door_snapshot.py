from __future__ import annotations

import hashlib

import pytest
import torch

from flightrl.puffer4_door_policy_contract import DoorPolicyArchitecture
from flightrl.puffer4_door_runtime import DoorPufferShadow, DoorPufferRuntime
from flightrl.puffer4_door_snapshot import load_fixed_door_checkpoint_snapshot


def test_checkpoint_snapshot_loads_the_exact_hashed_bytes(tmp_path) -> None:
    checkpoint = tmp_path / "door.bin"
    torch.save(DoorPufferRuntime(hidden_size=32).state_dict(), checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    snapshot = load_fixed_door_checkpoint_snapshot(checkpoint, digest)
    checkpoint.write_bytes(b"changed after snapshot")
    shadow = DoorPufferShadow.from_state_dict(
        snapshot.state_dict,
        architecture=DoorPolicyArchitecture(32, 1),
    )

    assert snapshot.sha256 == digest
    assert shadow.policy.encoder.fusion[0].weight.shape[0] == 32


def test_checkpoint_snapshot_rejects_bytes_changed_after_bundle_hash(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "door.bin"
    torch.save(DoorPufferRuntime(hidden_size=32).state_dict(), checkpoint)

    with pytest.raises(ValueError, match="snapshot SHA-256"):
        load_fixed_door_checkpoint_snapshot(checkpoint, "0" * 64)
