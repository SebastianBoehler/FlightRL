from .checkpoint_contract import (
    CHECKPOINT_CONTRACT_ID,
    CHECKPOINT_SCHEMA,
    build_checkpoint_payload,
    require_current_checkpoint,
)
from .env import SixDofCrazyflieEnv
from .geometry import AxisAlignedObstacle, BoxRoom
from .native import native_step, native_step_env
from .tasks import MULTITASK, TASKS, append_task_encoding, parse_task_spec

__all__ = [
    "BoxRoom",
    "AxisAlignedObstacle",
    "CHECKPOINT_CONTRACT_ID",
    "CHECKPOINT_SCHEMA",
    "MULTITASK",
    "TASKS",
    "SixDofCrazyflieEnv",
    "SixDofPolicy",
    "append_task_encoding",
    "build_checkpoint_payload",
    "checkpoint_tasks",
    "evaluate_checkpoint_policy",
    "evaluate_policy",
    "evaluate_teacher",
    "gate_status",
    "load_controller_from_checkpoint",
    "load_policy_from_checkpoint",
    "native_step",
    "native_step_env",
    "parse_task_spec",
    "require_current_checkpoint",
    "teacher_actions",
]


def __getattr__(name: str):
    evaluation_names = {
        "checkpoint_tasks",
        "evaluate_checkpoint_policy",
        "evaluate_policy",
        "evaluate_teacher",
        "load_controller_from_checkpoint",
        "load_policy_from_checkpoint",
    }
    policy_names = {"SixDofPolicy", "teacher_actions"}
    if name in evaluation_names:
        from . import evaluation

        return getattr(evaluation, name)
    if name in policy_names:
        from . import policies

        return getattr(policies, name)
    if name == "gate_status":
        from .gates import gate_status

        return gate_status
    raise AttributeError(name)
