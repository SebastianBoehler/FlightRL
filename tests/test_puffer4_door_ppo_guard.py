from __future__ import annotations

import pytest

from flightrl.puffer4_door_training import train_door_policy
from flightrl.puffer4_door_training_gates import (
    require_reset_safe_fixed_door_ppo,
)


def test_generic_recurrent_ppo_fails_closed_before_initializing_puffer() -> None:
    args = {"train": {}}

    with pytest.raises(
        RuntimeError,
        match="recurrent state is not masked at episode terminals",
    ):
        train_door_policy(
            args,
            object(),
            observability_checkpoint={},
            total_timesteps=1,
            bootstrap_updates=0,
            bootstrap_learning_rate=0.001,
            bootstrap_max_policy_rollin=0.0,
            log_interval=1,
        )

    assert args == {"train": {}}


def test_reset_aware_bc_dagger_budget_remains_enabled() -> None:
    require_reset_safe_fixed_door_ppo(0)
