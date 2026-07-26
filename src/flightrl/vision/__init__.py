from .contract import (
    VisionObservationBatchEncoder,
    VisionObservationConfig,
    VisionObservationEncoder,
    append_vision_observation,
)
from .action_dataset import (
    VisionActionDataset,
    VisionActionScale,
    load_aligned_vision_actions,
    phase_holdout_split,
)
__all__ = [
    "CompactVisionActionPolicy",
    "VisionActionDataset",
    "VisionActionPolicyMetadata",
    "VisionActionScale",
    "VisionObservationBatchEncoder",
    "VisionObservationConfig",
    "VisionObservationEncoder",
    "append_vision_observation",
    "load_aligned_vision_actions",
    "load_vision_action_policy",
    "phase_holdout_split",
    "save_vision_action_policy",
]


def __getattr__(name: str):
    policy_names = {
        "CompactVisionActionPolicy",
        "VisionActionPolicyMetadata",
        "load_vision_action_policy",
        "save_vision_action_policy",
    }
    if name not in policy_names:
        raise AttributeError(name)
    try:
        from . import action_policy
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise ModuleNotFoundError(
            "vision policy support requires PyTorch; install the training dependencies"
        ) from exc
    return getattr(action_policy, name)
