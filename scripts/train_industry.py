"""Train both embodiment controls from measured visual features; freeze on validation."""

import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from flightrl.robotics.pair_policy import PairNetwork, CONTRACT, LIMITS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(781)
    torch.set_num_threads(2)
    device = "mps"
    sets = []
    for split in ("train", "validation"):
        d = np.load(args.data / f"{split}.npz")
        sets.append(
            (
                torch.tensor(d["features"], device=device),
                torch.tensor(d["action"] / LIMITS, device=device),
            )
        )
    model = PairNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    best = float("inf")
    history = []
    started = time.perf_counter()
    for epoch in range(350):
        model.train()
        order = torch.randperm(len(sets[0][0]), device=device)
        for indices in order.split(256):
            prediction = model(sets[0][0][indices])
            loss = ((prediction - sets[0][1][indices]) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            score = float(((model(sets[1][0]) - sets[1][1]) ** 2).mean())
        history.append(dict(epoch=epoch, validation_mse=score))
        if score < best:
            best = score
            torch.save(
                dict(
                    contract=CONTRACT,
                    epoch=epoch,
                    model={k: v.cpu() for k, v in model.state_dict().items()},
                ),
                args.output / "actor.pt",
            )
        if epoch % 50 == 0:
            print(history[-1], flush=True)
    checkpoint = torch.load(args.output / "actor.pt", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        mae = (
            ((model(sets[1][0]) - sets[1][1]) * torch.tensor(LIMITS, device=device))
            .abs()
            .mean(0)
            .cpu()
            .tolist()
        )
    report = dict(
        history=history,
        selected_epoch=checkpoint["epoch"],
        validation_action_mae=mae,
        parameters=sum(p.numel() for p in model.parameters()),
        wall_s=time.perf_counter() - started,
        actor_sha256=hashlib.sha256(
            (args.output / "actor.pt").read_bytes()
        ).hexdigest(),
        dataset_sha256={
            s: hashlib.sha256((args.data / f"{s}.npz").read_bytes()).hexdigest()
            for s in ("train", "validation")
        },
    )
    (args.output / "training.json").write_text(json.dumps(report, indent=2))
    print("Frozen mixed-robot policy saved", flush=True)


if __name__ == "__main__":
    main()
