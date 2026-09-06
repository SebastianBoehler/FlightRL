"""Freeze a sensor-only policy experiment and train RGB-D imitation weights."""

import json
import time
from pathlib import Path
import numpy as np
import torch
from flightrl.fleet.camera_policy.data import dataset
from flightrl.fleet.camera_policy.network import Network, tensors

root = Path("artifacts/camera-control-20260906")
root.mkdir(exist_ok=True)
if (root / "actor.pt").exists():
    raise FileExistsError("Refusing to overwrite a trained experiment")
(root / "plan.json").write_text(
    json.dumps(
        {
            "train_seed": 3100,
            "validation_seed": 3101,
            "development_seeds": [3200, 3201, 3202],
            "test_seeds": list(range(3300, 3312)),
            "demo_seed": 3300,
            "actor_inputs": [
                "RGB",
                "depth",
                "body velocity estimate",
                "gravity direction",
                "gyro",
                "role flag",
                "peer visual reports and ages",
            ],
            "forbidden_inputs": [
                "position",
                "target coordinates",
                "route",
                "obstacle geometry",
                "simulator visibility labels",
            ],
            "scope": "Role-conditioned visual beacon approach; direct CTBR; ideal sensor estimates; no real person detector",
        },
        indent=2,
    )
)
started = time.perf_counter()
torch.manual_seed(31)
torch.set_num_threads(2)
print("Generating training camera observations", flush=True)
p, a, f = dataset(3100, 18000)
vp, va, vf = dataset(3101, 2000)
np.savez_compressed(
    root / "validation.npz",
    rgb=vp.rgb,
    depth=vp.depth,
    proprio=vp.proprio,
    role=vp.role,
    messages=vp.messages,
    actions=va,
    found=vf,
)
device = "mps"
images, states = tensors(p, device)
vi, vs = tensors(vp, device)
labels = torch.tensor(a, device=device)
found = torch.tensor(f, device=device)
vl = torch.tensor(va, device=device)
vfound = torch.tensor(vf, device=device)
model = Network().to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
best = float("inf")
for epoch in range(90):
    for ix in np.array_split(np.random.default_rng(epoch).permutation(len(a)), 72):
        output = model(images[ix], states[ix])
        loss = (
            output[:, :4] - labels[ix]
        ).square().mean() + 0.05 * torch.nn.functional.binary_cross_entropy_with_logits(
            output[:, 4], found[ix]
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        output = model(vi, vs)
        mse = float((output[:, :4] - vl).square().mean().cpu())
        bce = float(
            torch.nn.functional.binary_cross_entropy_with_logits(
                output[:, 4], vfound
            ).cpu()
        )
    score = mse + 0.05 * bce
    if score < best:
        best = score
        torch.save(
            {
                "model": {k: v.cpu() for k, v in model.state_dict().items()},
                "scope": "Raw RGB-D + ideal body sensors + role + delayed visual messages; direct CTBR",
            },
            root / "actor.pt",
        )
    if epoch % 10 == 0:
        print(epoch, mse, bce, flush=True)
(root / "training.json").write_text(
    json.dumps(
        {
            "train_images": len(a),
            "validation_images": len(va),
            "epochs": 90,
            "best_validation_loss": best,
            "last_validation_action_mse": mse,
            "last_validation_report_bce": bce,
            "wall_seconds": time.perf_counter() - started,
            "parameters": sum(x.numel() for x in model.parameters()),
        },
        indent=2,
    )
)
print("Training finished", flush=True)
