from __future__ import annotations

from pathlib import Path

import gymnasium
import numpy as np
import torch

from flightrl.policy import NativeFlightPolicy, create_policy_for_checkpoint


class DummyEnv:
    single_observation_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
    single_action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


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
