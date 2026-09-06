"""Learn route-cost bids; assignment and flight remain explicit algorithms."""

import numpy as np
import torch
from torch import nn
from flightrl.inspection.environments import environment_scene
from .routing import Routes


class BidNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def samples(seeds, pairs=500):
    data = []
    for seed in seeds:
        route = Routes(environment_scene("forest", seed))
        sites = route.sites(seed, 40)
        rng = np.random.default_rng(seed)
        for _ in range(pairs):
            a, b = sites[rng.choice(len(sites), 2, replace=False)]
            data.append((route.features(a, b), route.length(a, b) / 8))
    x, y = zip(*data)
    return torch.tensor(np.array(x)), torch.tensor(y, dtype=torch.float32)


def train(path, seeds, validation_seeds):
    torch.manual_seed(17)
    torch.set_num_threads(2)
    x, y = samples(seeds)
    vx, vy = samples(validation_seeds, 200)
    model = BidNetwork()
    opt = torch.optim.Adam(model.parameters(), lr=0.002)
    best = float("inf")
    best_state = None
    for epoch in range(100):
        for ix in torch.randperm(len(x)).split(256):
            loss = (model(x[ix]) - y[ix]).square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            mae = float((model(vx) - vy).abs().mean() * 8)
        if mae < best:
            best = mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    torch.save(
        {
            "model": best_state,
            "scope": "route-cost bids, known-map planner, supervised labels",
        },
        path,
    )
    return {
        "samples": len(x),
        "validation_samples": len(vx),
        "validation_mae_m": best,
        "epochs": 100,
        "parameters": sum(p.numel() for p in model.parameters()),
    }


class LearnedBids:
    def __init__(self, path):
        self.model = BidNetwork()
        self.model.load_state_dict(torch.load(path, weights_only=True)["model"])
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, route, a, b):
        return float(self.model(torch.tensor(route.features(a, b))) * 8)
