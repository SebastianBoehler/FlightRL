from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pufferlib


class FlightPolicy(nn.Module):
    def __init__(self, env, hidden_size: int = 128):
        super().__init__()
        obs_dim = int(np.prod(env.single_observation_space.shape))
        act_dim = int(np.prod(env.single_action_space.shape))
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.decoder_mean = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, act_dim), std=0.01)
        self.decoder_logstd = nn.Parameter(torch.zeros(1, act_dim))
        self.value = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def encode_observations(self, observations, state=None):
        del state
        return self.encoder(observations.view(observations.shape[0], -1).float())

    def decode_actions(self, hidden):
        mean = self.decoder_mean(hidden)
        std = torch.exp(self.decoder_logstd.expand_as(mean))
        return torch.distributions.Normal(mean, std), self.value(hidden)

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations, state=state)
        return self.decode_actions(hidden)

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state=state)
