from types import SimpleNamespace

import torch

from flightrl.mujoco.semantic_puffer_trainer import StatefulSemanticPuffeRL


class _Profile:
    ROLLOUT = 0
    EVAL_GPU = 1
    EVAL_ENV = 2

    def mark(self, _index: int) -> None:
        pass

    def elapsed(self, _metric: int, _start: int, _end: int) -> None:
        pass


class _Policy:
    def __init__(self) -> None:
        self.input_states: list[float] = []

    def forward_eval(self, observations: torch.Tensor, state):
        self.input_states.append(float(state[0].item()))
        next_state = (state[0] + 1.0,)
        distribution = torch.distributions.Normal(
            torch.zeros((observations.shape[0], 1)),
            torch.ones((observations.shape[0], 1)),
        )
        return distribution, torch.zeros(observations.shape[0]), next_state


class _Vector:
    gpu = False

    def __init__(self, rewards: torch.Tensor, terminals: torch.Tensor) -> None:
        self.rewards = rewards
        self.terminals = terminals
        self.steps = 0

    def cpu_step(self, _actions_ptr: int) -> None:
        self.steps += 1
        self.rewards[:] = float(self.steps)
        self.terminals[:] = float(self.steps == 2)

    def log(self) -> dict:
        return {}


def _sample_logits(distribution, action=None):
    sampled = distribution.mean if action is None else action
    count = distribution.mean.shape[0]
    return sampled, torch.zeros(count), torch.zeros(count)


def test_stateful_rollout_carries_state_and_masks_boundary_terminal() -> None:
    horizon = 2
    observations = torch.zeros((1, 1))
    rewards = torch.zeros(1)
    terminals = torch.zeros(1)
    policy = _Policy()
    base = SimpleNamespace(
        config={"horizon": horizon},
        device="cpu",
        total_agents=1,
        gpu=False,
        profile=_Profile(),
        policy=policy,
        state=(torch.zeros((1, 1, 1)),),
        vec_obs=observations,
        vec_rewards=rewards,
        vec_terminals=terminals,
        observations=torch.zeros((horizon, 1, 1)),
        actions=torch.zeros((horizon, 1, 1)),
        logprobs=torch.zeros((horizon, 1)),
        rewards=torch.zeros((horizon, 1)),
        terminals=torch.zeros((horizon, 1)),
        values=torch.zeros((horizon, 1)),
        global_step=0,
    )
    base._vec = _Vector(rewards, terminals)
    module = SimpleNamespace(Profile=_Profile, sample_logits=_sample_logits)
    trainer = StatefulSemanticPuffeRL(base, module)

    trainer.rollouts()
    assert policy.input_states == [0.0, 1.0]
    assert float(base.state[0].item()) == 2.0
    assert torch.equal(base.rewards[:, 0], torch.tensor((1.0, 2.0)))
    assert float(base.terminals[1, 0].item()) == 1.0

    trainer.rollouts()
    assert policy.input_states[2] == 0.0
    assert float(trainer.rollout_initial_state[0].item()) == 0.0
    assert torch.equal(
        trainer.state_resets[:, 0],
        torch.zeros(horizon),
    )
