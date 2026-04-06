from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _obs_dim(env) -> int:
    return int(np.prod(env.single_observation_space.shape))


def _act_dim(env) -> int:
    return int(np.prod(env.single_action_space.shape))


def _resolve_device(device: str) -> str:
    if device == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return "cpu"


def _aligned_slice(weights: np.ndarray, index: int, count: int) -> tuple[np.ndarray, int]:
    end = index + count
    if end > len(weights):
        raise ValueError("checkpoint is smaller than the expected policy layout")
    chunk = weights[index:end]
    return chunk, (end + 7) & ~7


class DefaultEncoder(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Linear(obs_size, hidden_size)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.encoder(observations.view(observations.shape[0], -1).float())


class DefaultDecoder(nn.Module):
    def __init__(self, action_dim: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.decoder_mean = nn.Linear(hidden_size, action_dim)
        self.decoder_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.value_function = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.distributions.Normal, torch.Tensor]:
        mean = self.decoder_mean(hidden)
        logstd = self.decoder_logstd.expand_as(mean)
        logits = torch.distributions.Normal(mean, torch.exp(logstd))
        values = self.value_function(hidden)
        return logits, values


class MLP(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int = 1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.GELU()])
        self.net = nn.Sequential(*layers)

    def initial_state(self, batch_size: int, device: torch.device | str) -> tuple[()]:
        del batch_size, device
        return ()

    def forward_eval(self, hidden: torch.Tensor, state: tuple[()]) -> tuple[torch.Tensor, tuple[()]]:
        del state
        return self.net(hidden), ()


class MinGRU(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int = 1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, 3 * hidden_size, bias=False) for _ in range(num_layers)]
        )

    def _g(self, value: torch.Tensor) -> torch.Tensor:
        return torch.where(value >= 0, value + 0.5, value.sigmoid())

    def _log_g(self, value: torch.Tensor) -> torch.Tensor:
        return torch.where(value >= 0, (F.relu(value) + 0.5).log(), -F.softplus(-value))

    def _highway(self, x: torch.Tensor, out: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
        gate = proj.sigmoid()
        return gate * out + (1.0 - gate) * x

    def initial_state(self, batch_size: int, device: torch.device | str) -> tuple[torch.Tensor]:
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),)

    def forward_eval(self, hidden: torch.Tensor, state: tuple[torch.Tensor]) -> tuple[torch.Tensor, tuple[torch.Tensor]]:
        current = hidden.unsqueeze(1)
        recurrent = state[0]
        state_out = []
        for idx, layer in enumerate(self.layers):
            projected = layer(current)
            inner, gate, highway = projected.chunk(3, dim=-1)
            out = torch.lerp(recurrent[idx : idx + 1].transpose(0, 1), self._g(inner), gate.sigmoid())
            current = self._highway(current, out, highway)
            state_out.append(out[:, -1:])
        return current.squeeze(1), (torch.stack(state_out, 0).squeeze(2),)

    def forward_train(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            inner, gate, highway = layer(hidden).chunk(3, dim=-1)
            log_coeffs = -F.softplus(gate)
            log_values = -F.softplus(-gate) + self._log_g(inner)
            accum = log_coeffs.cumsum(dim=1)
            out = (accum + (log_values - accum).logcumsumexp(dim=1)).exp()
            hidden = self._highway(hidden, out, highway)
        return hidden


class FlightPolicy(nn.Module):
    def __init__(self, env, hidden_size: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        self.encoder = DefaultEncoder(_obs_dim(env), hidden_size)
        self.network = MLP(hidden_size, num_layers=num_layers)
        self.decoder = DefaultDecoder(_act_dim(env), hidden_size)

    def initial_state(self, batch_size: int, device: torch.device | str) -> tuple[()]:
        return self.network.initial_state(batch_size, device)

    def forward_eval(
        self,
        observations: torch.Tensor,
        state: tuple[()] | None = None,
    ) -> tuple[torch.distributions.Normal, torch.Tensor, tuple[()]]:
        hidden = self.encoder(observations)
        hidden, state = self.network.forward_eval(hidden, () if state is None else state)
        logits, values = self.decoder(hidden)
        return logits, values, state


class NativeFlightPolicy(nn.Module):
    def __init__(self, env, hidden_size: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        self.action_dim = _act_dim(env)
        self.encoder = nn.Linear(_obs_dim(env), hidden_size, bias=False)
        self.network = MinGRU(hidden_size, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_size, self.action_dim + 1, bias=False)
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))

    def initial_state(self, batch_size: int, device: torch.device | str) -> tuple[torch.Tensor]:
        return self.network.initial_state(batch_size, device)

    def forward_eval(
        self,
        observations: torch.Tensor,
        state: tuple[torch.Tensor] | None = None,
    ) -> tuple[torch.distributions.Normal, torch.Tensor, tuple[torch.Tensor]]:
        batch = observations.shape[0]
        hidden = self.encoder(observations.view(batch, -1).float())
        hidden, next_state = self.network.forward_eval(
            hidden,
            self.initial_state(batch, observations.device) if state is None else state,
        )
        decoded = self.decoder(hidden)
        mean = decoded[:, : self.action_dim]
        values = decoded[:, self.action_dim :]
        logits = torch.distributions.Normal(mean, torch.exp(self.log_std).expand_as(mean))
        return logits, values, next_state


def create_policy_for_checkpoint(
    env,
    checkpoint_path: str | Path | None,
    *,
    hidden_size: int = 128,
    num_layers: int = 2,
    device: str = "cpu",
) -> nn.Module:
    target_device = _resolve_device(device)
    path = None if checkpoint_path is None else Path(checkpoint_path)
    if path is None:
        return FlightPolicy(env, hidden_size=hidden_size, num_layers=num_layers).to(target_device)

    torch_policy = FlightPolicy(env, hidden_size=hidden_size, num_layers=num_layers)
    if _try_load_torch_checkpoint(torch_policy, path, target_device):
        return torch_policy.to(target_device)

    native_policy = NativeFlightPolicy(env, hidden_size=hidden_size, num_layers=num_layers)
    _load_native_checkpoint(native_policy, path)
    return native_policy.to(target_device)


def _try_load_torch_checkpoint(policy: nn.Module, checkpoint_path: Path, device: str) -> bool:
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
    except Exception as exc:
        if checkpoint_path.suffix == ".bin":
            return False
        raise RuntimeError("failed to load checkpoint as a torch policy") from exc
    if not isinstance(state_dict, dict):
        return False
    policy.load_state_dict({key.replace("module.", ""): value for key, value in state_dict.items()})
    return True


def _load_native_checkpoint(policy: nn.Module, checkpoint_path: Path) -> None:
    if not isinstance(policy, NativeFlightPolicy):
        raise TypeError("native checkpoints require NativeFlightPolicy")
    weights = np.fromfile(checkpoint_path, dtype=np.float32)
    if weights.size == 0:
        raise RuntimeError(f"native checkpoint is empty: {checkpoint_path}")

    hidden_size = policy.encoder.weight.shape[0]
    action_dim = policy.action_dim
    index = 0

    encoder, index = _aligned_slice(weights, index, policy.encoder.weight.numel())
    decoder, index = _aligned_slice(weights, index, policy.decoder.weight.numel())
    log_std, index = _aligned_slice(weights, index, action_dim)

    with torch.no_grad():
        policy.encoder.weight.copy_(torch.from_numpy(encoder.reshape(policy.encoder.weight.shape)))
        policy.decoder.weight.copy_(torch.from_numpy(decoder.reshape(policy.decoder.weight.shape)))
        policy.log_std.copy_(torch.from_numpy(log_std))
        for layer in policy.network.layers:
            chunk, index = _aligned_slice(weights, index, layer.weight.numel())
            layer.weight.copy_(torch.from_numpy(chunk.reshape(3 * hidden_size, hidden_size)))
