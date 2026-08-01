from __future__ import annotations

import torch

import flightrl.puffer4_door_checkpoint as door_checkpoint
from flightrl.puffer4_door_checkpoint import (
    load_door_perception_state,
    migrate_door_policy_state,
)
from flightrl.puffer4_door_runtime import DoorPufferRuntime


def test_migration_preserves_old_features_and_zero_initializes_evidence() -> None:
    policy = DoorPufferRuntime(hidden_size=32)
    source = {key: value.clone() for key, value in policy.state_dict().items()}
    fusion = source["encoder.fusion.0.weight"]
    source["encoder.fusion.0.weight"] = torch.cat(
        (fusion[:, :165], fusion[:, 170:174]),
        dim=1,
    )

    result = migrate_door_policy_state(policy, source)
    migrated = policy.state_dict()["encoder.fusion.0.weight"]

    assert result["migrated_tensors"] == 1
    torch.testing.assert_close(migrated[:, :165], fusion[:, :165])
    assert torch.count_nonzero(migrated[:, 165:170]) == 0
    torch.testing.assert_close(migrated[:, 170:174], fusion[:, 170:174])


def test_perception_warmstart_leaves_control_network_fresh() -> None:
    policy = DoorPufferRuntime(hidden_size=32)
    source = DoorPufferRuntime(hidden_size=32)
    with torch.no_grad():
        source.encoder.visual[0].weight.fill_(0.25)
        source.encoder.grounder.encoder[0].weight.fill_(0.5)
        source.encoder.fusion[0].weight.fill_(0.75)
    original_fusion = policy.encoder.fusion[0].weight.detach().clone()

    result = load_door_perception_state(policy, source.state_dict())

    assert result["loaded_tensors"] > 0
    torch.testing.assert_close(
        policy.encoder.visual[0].weight,
        source.encoder.visual[0].weight,
    )
    torch.testing.assert_close(
        policy.encoder.grounder.encoder[0].weight,
        source.encoder.grounder.encoder[0].weight,
    )
    torch.testing.assert_close(policy.encoder.fusion[0].weight, original_fusion)


def test_fresh_control_initialization_is_paired_by_seed() -> None:
    initializer = getattr(door_checkpoint, "initialize_door_policy", None)
    assert initializer is not None
    source = DoorPufferRuntime(hidden_size=32).state_dict()

    first, _ = initializer(
        lambda: DoorPufferRuntime(hidden_size=32),
        source,
        seed=23,
        fresh_control=True,
    )
    second, _ = initializer(
        lambda: DoorPufferRuntime(hidden_size=32),
        source,
        seed=23,
        fresh_control=True,
    )

    for key in (
        "encoder.fusion.0.weight",
        "network.layers.0.weight",
        "decoder.decoder_mean.weight",
    ):
        torch.testing.assert_close(first.state_dict()[key], second.state_dict()[key])
