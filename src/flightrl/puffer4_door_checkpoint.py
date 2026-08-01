from __future__ import annotations

from collections.abc import Callable

import torch


_VISUAL_FEATURES = 12 * 3 * 4
_GROUNDER_FEATURES = 4


def initialize_door_policy(
    policy_factory: Callable[[], object],
    source: dict,
    *,
    seed: int,
    fresh_control: bool,
):
    torch.manual_seed(seed)
    policy = policy_factory()
    migration = (
        load_door_perception_state(policy, source)
        if fresh_control
        else migrate_door_policy_state(policy, source)
    )
    return policy, migration


def migrate_door_policy_state(policy, source: dict) -> dict[str, int]:
    """Load an older door actor while appending zero-weight evidence inputs."""
    current = policy.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    migrated = 0
    skipped = 0
    for key, value in source.items():
        target = current.get(key)
        if target is None:
            skipped += 1
        elif target.shape == value.shape:
            loaded[key] = value
        elif key == "encoder.fusion.0.weight":
            loaded[key] = _migrate_fusion_weight(value, target)
            migrated += 1
        else:
            skipped += 1
    policy.load_state_dict(loaded, strict=False)
    return {
        "loaded_tensors": len(loaded) - migrated,
        "migrated_tensors": migrated,
        "skipped_tensors": skipped,
    }


def load_door_perception_state(policy, source: dict) -> dict[str, int]:
    prefixes = ("encoder.visual.", "encoder.grounder.")
    current = policy.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key.startswith(prefixes)
        and key in current
        and current[key].shape == value.shape
    }
    policy.load_state_dict(compatible, strict=False)
    return {
        "loaded_tensors": len(compatible),
        "skipped_tensors": len(source) - len(compatible),
    }


def _migrate_fusion_weight(
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("door fusion migration requires matrices")
    if source.shape[0] != target.shape[0]:
        raise ValueError("door fusion hidden width changed")
    old_proprio = source.shape[1] - _VISUAL_FEATURES - _GROUNDER_FEATURES
    new_proprio = target.shape[1] - _VISUAL_FEATURES - _GROUNDER_FEATURES
    if old_proprio <= 0 or new_proprio < old_proprio:
        raise ValueError("door fusion proprio contract cannot be migrated")
    migrated = torch.zeros_like(target)
    old_grounder = _VISUAL_FEATURES + old_proprio
    new_grounder = _VISUAL_FEATURES + new_proprio
    migrated[:, :old_grounder] = source[:, :old_grounder]
    migrated[:, new_grounder:] = source[:, old_grounder:]
    return migrated
