"""Direct learned collective-thrust/body-rate controller, given a 3-D waypoint."""

import numpy as np
import torch
from torch import nn
from flightrl import _binding
from flightrl.fleet.vehicles import VEHICLES


def observation(delta, velocity, quaternion, heading):
    q = quaternion
    w, x, y, z = q.T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    error = heading - yaw
    return np.column_stack(
        (
            np.clip(delta, -3, 3) / 3,
            velocity,
            roll,
            pitch,
            np.sin(yaw),
            np.cos(yaw),
            np.sin(error),
            np.cos(error),
        )
    ).astype(np.float32)


def teacher(delta, velocity, quaternion, heading):
    w, x, y, z = quaternion.T
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(yaw), np.sin(yaw)
    desired = np.clip(1.3 * delta - 0.15 * velocity, -0.6, 0.6)
    error = (heading - yaw + np.pi) % (2 * np.pi) - np.pi
    command = np.column_stack(
        (
            (c * desired[:, 0] + s * desired[:, 1]) / 0.7,
            (-s * desired[:, 0] + c * desired[:, 1]) / 0.7,
            desired[:, 2] / 0.4,
            np.clip(error * 0.6, -0.5, 0.5),
        )
    ).astype(np.float32)
    physics = np.repeat(VEHICLES["fpv"].physics()[None], len(delta), axis=0)
    output = np.empty((len(delta), 4), np.float32)
    _binding.sixdof_setpoint_actions(
        velocity, quaternion, command, physics, output, 0.7, 0.4, 2.5, 6.0, 3.0
    )
    return output


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 96), nn.Tanh(), nn.Linear(96, 96), nn.Tanh(), nn.Linear(96, 4)
        )

    def forward(self, x):
        return self.net(x)


def samples(seed, count):
    rng = np.random.default_rng(seed)
    delta = rng.uniform(-2, 2, (count, 3)).astype(np.float32)
    velocity = rng.uniform(-0.8, 0.8, (count, 3)).astype(np.float32)
    angles = rng.uniform([-0.25, -0.25, -np.pi], [0.25, 0.25, np.pi], (count, 3))
    # Include near-hover and fine-positioning states in equal measure.
    delta[: count // 2] *= 0.15
    velocity[: count // 2] *= 0.25
    angles[: count // 2, :2] *= 0.3
    r, p, y = (angles / 2).T
    q = np.column_stack(
        (
            np.cos(r) * np.cos(p) * np.cos(y) + np.sin(r) * np.sin(p) * np.sin(y),
            np.sin(r) * np.cos(p) * np.cos(y) - np.cos(r) * np.sin(p) * np.sin(y),
            np.cos(r) * np.sin(p) * np.cos(y) + np.sin(r) * np.cos(p) * np.sin(y),
            np.cos(r) * np.cos(p) * np.sin(y) - np.sin(r) * np.sin(p) * np.cos(y),
        )
    ).astype(np.float32)
    heading = rng.uniform(-np.pi, np.pi, count)
    return observation(delta, velocity, q, heading), teacher(
        delta, velocity, q, heading
    )


def train(path):
    torch.set_num_threads(2)
    torch.manual_seed(44)
    x, y = map(torch.tensor, samples(44, 40000))
    vx, vy = map(torch.tensor, samples(45, 4000))
    model = Network()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best = float("inf")
    state = None
    for epoch in range(150):
        for ix in torch.randperm(len(x)).split(512):
            loss = ((model(x[ix]) - y[ix]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            validation = float(((model(vx) - vy) ** 2).mean())
        if validation < best:
            best = validation
            state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 30 == 0:
            print("flight epoch", epoch, validation, flush=True)
    torch.save(
        {
            "model": state,
            "scope": "Direct collective thrust and body rates; waypoint-conditioned proprioceptive imitation",
        },
        path,
    )
    return {
        "training_states": len(x),
        "validation_states": len(vx),
        "validation_mse": best,
        "parameters": sum(p.numel() for p in model.parameters()),
        "epochs": 150,
    }


class Policy:
    def __init__(self, path):
        model = Network()
        model.load_state_dict(torch.load(path, weights_only=True)["model"])
        model.eval()
        self.layers = [
            (layer.weight.detach().numpy().T, layer.bias.detach().numpy())
            for layer in model.net
            if isinstance(layer, nn.Linear)
        ]

    def __call__(self, delta, velocity, q, heading):
        x = observation(delta, velocity, q, heading)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = np.tanh(x)
        return np.ascontiguousarray(np.clip(x, -1, 1), np.float32)
