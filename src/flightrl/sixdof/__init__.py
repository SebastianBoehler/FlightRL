from .env import SixDofCrazyflieEnv
from .geometry import AxisAlignedObstacle, BoxRoom
from .native import native_step, native_step_env
from .tasks import MULTITASK, TASKS, append_task_encoding, parse_task_spec

__all__ = [
    "BoxRoom",
    "AxisAlignedObstacle",
    "MULTITASK",
    "TASKS",
    "SixDofCrazyflieEnv",
    "SixDofPolicy",
    "append_task_encoding",
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
    "teacher_actions",
]


def __getattr__(name: str):
    evaluation_names = {
        "checkpoint_tasks",
        "evaluate_checkpoint_policy",
        "evaluate_policy",
        "evaluate_teacher",
        "gate_status",
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
    raise AttributeError(name)
