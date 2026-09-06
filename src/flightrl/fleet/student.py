"""Shared local visual policy conditioned on vehicle, mission and peer messages."""

import numpy as np
import torch
from torch import nn
from flightrl.inspection.student import image_tensor


class FleetNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 12, 5, 3),
            nn.ReLU(),
            nn.Conv2d(12, 16, 3, 2),
            nn.ReLU(),
            nn.AvgPool2d((3, 4)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + 33, 64), nn.ReLU(), nn.Linear(64, 4), nn.Tanh()
        )

    def forward(self, image, state):
        return self.head(torch.cat((self.encoder(image), state), -1))


def train(data, path, epochs=12):
    torch.manual_seed(0)
    torch.set_num_threads(2)
    model = FleetNetwork().to("mps")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    rgb, depth, state, target = map(np.stack, zip(*data))
    images = image_tensor(rgb, depth, "mps")
    states = torch.tensor(state, device="mps")
    targets = torch.tensor(target, device="mps")
    generator = np.random.default_rng(0)
    for _ in range(epochs):
        for ix in np.array_split(
            generator.permutation(len(data)), max(1, len(data) // 128)
        ):
            pred = model(images[ix], states[ix])
            loss = (pred - targets[ix]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(
        {
            "model": {k: v.cpu() for k, v in model.state_dict().items()},
            "scope": "imitation learning, native analytic camera",
        },
        path,
    )
    return {
        "samples": len(data),
        "epochs": epochs,
        "final_batch_loss": float(loss.detach().cpu()),
        "parameters": sum(p.numel() for p in model.parameters()),
    }


class FleetPolicy:
    def __init__(self, path):
        self.model = FleetNetwork()
        self.model.load_state_dict(torch.load(path, weights_only=True)["model"])
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, rgb, depth, state):
        return self.model(
            image_tensor(rgb[None], depth[None], "cpu"), torch.tensor(state)[None]
        )[0].numpy()
