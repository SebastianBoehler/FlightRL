from __future__ import annotations

from math import isfinite
from pathlib import Path
import re


def edge_execution_provenance(
    execution_policy: object,
    checkpoint_identity: object,
    *,
    split: object,
    agents: object,
    student_fraction: object = None,
    mix_seed: object = None,
) -> dict:
    if execution_policy not in {"privileged_teacher", "dagger_student"}:
        raise ValueError("edge dataset execution policy is unsupported")
    if execution_policy == "dagger_student":
        if split != "train":
            raise ValueError("edge DAgger data is restricted to the train split")
        _validate_checkpoint_identity(checkpoint_identity)
        if type(agents) is not int or agents <= 0:
            raise ValueError("edge DAgger agents must be positive")
        if (
            isinstance(student_fraction, bool)
            or not isinstance(student_fraction, (int, float))
            or not isfinite(float(student_fraction))
            or not 0.0 < float(student_fraction) <= 1.0
        ):
            raise ValueError("edge DAgger student fraction must be in (0, 1]")
        student_agents = round(float(student_fraction) * agents)
        if abs(float(student_fraction) * agents - student_agents) > 1.0e-12:
            raise ValueError("edge DAgger student fraction must select exact agents")
        if type(mix_seed) is not int or not 0 <= mix_seed < 2**32:
            raise ValueError("edge DAgger execution mix seed must be uint32")
        student = student_agents / agents
        teacher = 1.0 - student
        identity = dict(checkpoint_identity)
        schedule = "fixed_per_agent_sha256_rank_v1"
    else:
        provenance = (checkpoint_identity, student_fraction, mix_seed)
        if any(value is not None for value in provenance):
            raise ValueError("edge teacher data cannot bind checkpoint provenance")
        teacher, student, identity = 1.0, 0.0, None
        schedule = "privileged_teacher"
    return {
        "execution_policy": execution_policy,
        "execution_checkpoint_identity": identity,
        "execution_mix": {
            "teacher": teacher,
            "student": student,
            "schedule": schedule,
            "seed": mix_seed,
        },
    }


def _validate_checkpoint_identity(identity: object) -> None:
    if not isinstance(identity, dict) or set(identity) != {"path", "sha256"}:
        raise ValueError("edge DAgger execution checkpoint identity is invalid")
    path = identity["path"]
    digest = identity["sha256"]
    if (
        not isinstance(path, str)
        or not path
        or not Path(path).is_absolute()
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise ValueError("edge DAgger execution checkpoint identity is invalid")
