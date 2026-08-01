from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import torch

from flightrl.sixdof import (
    CHECKPOINT_CONTRACT_ID,
    SixDofPolicy,
    build_checkpoint_payload,
    load_policy_from_checkpoint,
    require_current_checkpoint,
)
from flightrl.sixdof.dagger import collect_policy_dataset


ROOT = Path(__file__).resolve().parents[1]


def test_current_checkpoint_roundtrip_has_explicit_contract() -> None:
    checkpoint = policy_checkpoint()

    metadata = require_current_checkpoint(checkpoint)
    model = load_policy_from_checkpoint(checkpoint)

    assert checkpoint["checkpoint_contract"] == CHECKPOINT_CONTRACT_ID
    assert metadata.tasks == ("position_yaw",)
    assert model.input_dim == 28


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("checkpoint_schema", None),
        ("checkpoint_contract", "retired-sixdof-contract"),
    ),
)
def test_checkpoint_loader_rejects_missing_or_wrong_identity(
    field: str,
    value: object,
) -> None:
    checkpoint = policy_checkpoint()
    if value is None:
        checkpoint.pop(field)
    else:
        checkpoint[field] = value

    with pytest.raises(ValueError, match="legacy checkpoints are rejected"):
        load_policy_from_checkpoint(checkpoint)


def test_checkpoint_loader_does_not_infer_missing_semantics() -> None:
    checkpoint = policy_checkpoint()
    checkpoint.pop("observation_mode")

    with pytest.raises(ValueError, match="observation mode"):
        load_policy_from_checkpoint(checkpoint)


def test_dagger_rejects_uncontracted_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"state_dict": SixDofPolicy(hidden_size=16).state_dict()}, checkpoint)

    with pytest.raises(ValueError, match="legacy checkpoints are rejected"):
        collect_policy_dataset(
            checkpoint_path=checkpoint,
            task_spec=None,
            num_envs=2,
            steps=1,
            seed=7,
            use_native_step=False,
        )


def test_ppo_init_rejects_task_reinterpretation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "current.pt"
    torch.save(policy_checkpoint(), checkpoint)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_sixdof_ppo.py"),
            "--init-checkpoint",
            str(checkpoint),
            "--train-tasks",
            "obstacle_avoidance",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "cannot reinterpret" in result.stderr


def policy_checkpoint() -> dict:
    return build_checkpoint_payload(
        state_dict=SixDofPolicy(hidden_size=16).state_dict(),
        tasks=("position_yaw",),
        hidden_size=16,
    )
