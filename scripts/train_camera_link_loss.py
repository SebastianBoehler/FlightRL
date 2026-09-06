"""Address the observed v1 missing-message failure; reserve new test seeds."""

import json
import shutil
from pathlib import Path
import numpy as np
import torch
from flightrl.fleet.camera_policy.episode import run
from flightrl.fleet.camera_policy.network import Network, Policy, tensors
from flightrl.fleet.camera_policy.sensors import CameraPacket

root = Path("artifacts/camera-control-20260906")
second = Path("artifacts/camera-control-linkloss-20260906")
second.mkdir(exist_ok=False)
shutil.copyfile(root / "actor.pt", second / "initial.pt")
(second / "plan.json").write_text(
    json.dumps(
        {
            "training_seeds": list(range(3168, 3180)),
            "development_seeds": [3200, 3201, 3202],
            "test_seeds": list(range(3400, 3412)),
            "demo_seed": 3400,
            "purpose": "Teach missing-peer-message hold behavior after v1 failed link ablation; previous tests are not reused",
        },
        indent=2,
    )
)
data = []
policy = Policy(second / "initial.pt")
for seed in range(3168, 3180):
    for mode in [None, "no_messages"]:
        run(seed, ablation=mode, samples_out=data, ticks=300)
        run(seed, policy, ablation=mode, samples_out=data, ticks=300)
p = CameraPacket(
    *[
        np.concatenate([getattr(d[0], name) for d in data])
        for name in ["rgb", "depth", "proprio", "role", "messages"]
    ],
    0,
    0.0,
)
a = np.concatenate([d[1] for d in data])
f = np.concatenate([d[2] for d in data])
torch.manual_seed(33)
torch.set_num_threads(2)
images, states = tensors(p, "mps")
actions = torch.tensor(a, device="mps")
found = torch.tensor(f, device="mps")
model = Network().to("mps")
model.load_state_dict(torch.load(second / "initial.pt", weights_only=True)["model"])
opt = torch.optim.Adam(model.parameters(), lr=0.00015)
for epoch in range(20):
    for ix in np.array_split(
        np.random.default_rng(epoch).permutation(len(a)), max(1, len(a) // 256)
    ):
        out = model(images[ix], states[ix])
        loss = (
            out[:, :4] - actions[ix]
        ).square().mean() + 0.04 * torch.nn.functional.binary_cross_entropy_with_logits(
            out[:, 4], found[ix]
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    if epoch % 5 == 0:
        print(epoch, float(loss.detach().cpu()), flush=True)
torch.save(
    {
        "model": {k: v.cpu() for k, v in model.state_dict().items()},
        "scope": "RGB-D direct control with explicit absent peer messages in training",
    },
    second / "actor.pt",
)
p = Policy(second / "actor.pt")
results = {}
for mode in [None, "no_messages"]:
    for seed in [3200, 3201, 3202]:
        results[f"{mode}/{seed}"] = run(seed, p, mode)[0]["result"]
(second / "development.json").write_text(
    json.dumps({"training_images": len(a), "epochs": 20, "results": results}, indent=2)
)
print(results, flush=True)
