from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from flightrl.mujoco.semantic_imitation import collect_expert_sequence


class _StatePolicy:
    def initial_state(self, batch_size: int, device: str):
        return (torch.zeros(1, batch_size, 1, device=device),)

    def forward_eval(self, observations: torch.Tensor, state):
        next_state = (state[0] + 1.0,)
        action = torch.zeros((observations.shape[0], 4))
        return SimpleNamespace(mean=action), torch.zeros(observations.shape[0]), next_state


class _TerminatingDriver:
    total_agents = 1

    def __init__(self) -> None:
        self.observations = np.zeros((1, 2), dtype=np.float32)
        self.terminals = np.zeros(1, dtype=np.float32)
        self.steps = 0

    def expert_actions(self) -> np.ndarray:
        return np.zeros((1, 4), dtype=np.float32)

    def target_observed(self) -> np.ndarray:
        return np.zeros(1, dtype=bool)

    def target_visible(self) -> np.ndarray:
        return np.zeros(1, dtype=bool)

    def action_corridor_clearance(self) -> np.ndarray:
        return np.ones(1, dtype=np.float32)

    def cpu_step(self, _actions_ptr: int) -> None:
        self.steps += 1
        self.terminals[:] = self.steps == 2


def test_expert_sequence_carries_state_and_masks_terminal_agents() -> None:
    initial = (torch.full((1, 1, 1), 5.0),)

    sequence = collect_expert_sequence(
        _TerminatingDriver(),
        3,
        rollout_policy=_StatePolicy(),
        policy_state=initial,
    )

    assert sequence.start_state is not None
    assert sequence.next_state is not None
    assert float(sequence.start_state[0].item()) == 5.0
    assert float(sequence.next_state[0].item()) == 1.0
    assert sequence.start_state[0].data_ptr() != initial[0].data_ptr()
    assert sequence.next_state[0].grad_fn is None
    assert sequence.state_resets.tolist() == [[0.0, 0.0, 1.0]]
    assert sequence.corridor_clearance.tolist() == [[1.0, 1.0, 1.0]]
