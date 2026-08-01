from __future__ import annotations

from pathlib import Path

import gymnasium
import numpy as np
import torch

from flightrl.policy import MinGRU, NativeFlightPolicy, create_policy_for_checkpoint


class DummyEnv:
    single_observation_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
    single_action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


def test_mingru_stateful_parallel_scan_matches_stepwise_inference() -> None:
    torch.manual_seed(7)
    network = MinGRU(hidden_size=5, num_layers=2)
    inputs = torch.randn(3, 11, 5)
    state = (torch.rand(2, 3, 5),)

    parallel, parallel_state = network.forward_train_stateful(inputs, state)
    sequential = []
    sequential_state = state
    for step in range(inputs.shape[1]):
        output, sequential_state = network.forward_eval(
            inputs[:, step],
            sequential_state,
        )
        sequential.append(output)

    assert torch.allclose(
        parallel,
        torch.stack(sequential, dim=1),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        parallel_state[0],
        sequential_state[0],
        atol=1e-5,
        rtol=1e-5,
    )


def test_mingru_masked_training_matches_stepwise_episode_resets() -> None:
    torch.manual_seed(11)
    network = MinGRU(hidden_size=4, num_layers=2)
    inputs = torch.randn(3, 7, 4)
    initial_state = (torch.rand(2, 3, 4),)
    terminals = torch.zeros(3, 7)
    terminals[0, 3] = 1.0
    terminals[2, 1] = 1.0
    terminals[2, 5] = 1.0

    parallel, parallel_state = network.forward_train_stateful_masked(
        inputs,
        initial_state,
        terminals,
    )
    sequential = []
    sequential_state = initial_state
    for step in range(inputs.shape[1]):
        alive = (1.0 - terminals[:, step]).reshape(1, -1, 1)
        sequential_state = tuple(value * alive for value in sequential_state)
        output, sequential_state = network.forward_eval(
            inputs[:, step],
            sequential_state,
        )
        sequential.append(output)

    assert torch.allclose(
        parallel,
        torch.stack(sequential, dim=1),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        parallel_state[0],
        sequential_state[0],
        atol=1e-5,
        rtol=1e-5,
    )


def test_native_checkpoint_loader_reads_aligned_weights(tmp_path: Path) -> None:
    env = DummyEnv()
    hidden_size = 4
    num_layers = 2

    encoder = np.arange(hidden_size * 5, dtype=np.float32)
    decoder = np.arange((2 + 1) * hidden_size, dtype=np.float32) + 100
    log_std = np.arange(2, dtype=np.float32) + 200
    recurrent = [
        np.arange(3 * hidden_size * hidden_size, dtype=np.float32) + 300,
        np.arange(3 * hidden_size * hidden_size, dtype=np.float32) + 400,
    ]

    flat: list[np.ndarray] = []
    for chunk in (encoder, decoder, log_std, *recurrent):
        flat.append(chunk)
        pad = (-chunk.size) % 8
        if pad:
            flat.append(np.zeros(pad, dtype=np.float32))

    checkpoint = tmp_path / "policy.bin"
    np.concatenate(flat).tofile(checkpoint)

    policy = create_policy_for_checkpoint(
        env,
        checkpoint,
        hidden_size=hidden_size,
        num_layers=num_layers,
        device="cpu",
    )
    assert isinstance(policy, NativeFlightPolicy)
    assert torch.equal(policy.encoder.weight, torch.from_numpy(encoder.reshape(hidden_size, 5)))
    assert torch.equal(policy.decoder.weight, torch.from_numpy(decoder.reshape(3, hidden_size)))
    assert torch.equal(policy.log_std, torch.from_numpy(log_std))
    for idx, layer in enumerate(policy.network.layers):
        expected = torch.from_numpy(recurrent[idx].reshape(3 * hidden_size, hidden_size))
        assert torch.equal(layer.weight, expected)
