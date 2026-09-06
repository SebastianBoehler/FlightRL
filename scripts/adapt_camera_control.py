"""DAgger on training episodes only; retain the initial failed camera actor."""

import json
import shutil
from pathlib import Path
import numpy as np
import torch
from flightrl.fleet.camera_policy.episode import run
from flightrl.fleet.camera_policy.network import Network, Policy, tensors
from flightrl.fleet.camera_policy.sensors import CameraPacket

root = Path("artifacts/camera-control-20260906")
if (
    (root / "actor-initial.pt").exists()
    or any(root.glob("adaptation*.json"))
    or any(root.glob("actor-round-*.pt"))
):
    raise FileExistsError(f"Adaptation evidence already exists in {root}")
shutil.copyfile(root / "actor.pt", root / "actor-initial.pt")
(root / "adaptation-plan.json").write_text(
    json.dumps(
        {
            "rounds": 3,
            "teacher_seeds": list(range(3120, 3144)),
            "learner_seeds": list(range(3144, 3168)),
            "selection": "fixed three rounds; no held-out seed access",
            "method": "DAgger with camera/depth visual-servo teacher",
        },
        indent=2,
    )
)
torch.set_num_threads(2)
torch.manual_seed(32)
data = []
for seed in range(3120, 3144):
    run(seed, samples_out=data)
metrics = []
for round in range(3):
    policy = Policy(root / "actor.pt")
    for seed in range(3144, 3168):
        run(seed, policy, samples_out=data)
    p = CameraPacket(
        *[
            np.concatenate([getattr(item[0], name) for item in data])
            for name in ["rgb", "depth", "proprio", "role", "messages"]
        ],
        0,
        0.0,
    )
    target = np.concatenate([item[1] for item in data])
    detected = np.concatenate([item[2] for item in data])
    images, states = tensors(p, "mps")
    actions = torch.tensor(target, device="mps")
    found = torch.tensor(detected, device="mps")
    model = Network().to("mps")
    model.load_state_dict(torch.load(root / "actor.pt", weights_only=True)["model"])
    opt = torch.optim.Adam(model.parameters(), lr=0.0003)
    for epoch in range(18):
        for ix in np.array_split(
            np.random.default_rng(round * 100 + epoch).permutation(len(target)),
            max(1, len(target) // 256),
        ):
            output = model(images[ix], states[ix])
            loss = (
                (output[:, :4] - actions[ix]).square().mean()
                + 0.04
                * torch.nn.functional.binary_cross_entropy_with_logits(
                    output[:, 4], found[ix]
                )
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
    torch.save(
        {
            "model": {k: v.cpu() for k, v in model.state_dict().items()},
            "scope": "Sensor-only DAgger CTBR actor",
        },
        root / "actor.pt",
    )
    shutil.copyfile(root / "actor.pt", root / f"actor-round-{round}.pt")
    del images, states, actions, found, model, opt
    policy = Policy(root / "actor.pt")
    outcome = [run(seed, policy)[0]["result"] for seed in [3200, 3201, 3202]]
    metrics.append(
        {"round": round, "training_images": len(target), "development": outcome}
    )
    (root / "adaptation.json").write_text(json.dumps(metrics, indent=2))
    print(metrics[-1], flush=True)
