"""Fit the industrial actor to rendered observations, selecting only on validation."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from flightrl.robotics.policy import CONTRACT, LIMITS, Network, images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(711)
    torch.set_num_threads(4)
    device = torch.device("mps")
    datasets = []
    for split in ("train", "validation"):
        data = np.load(args.data / f"{split}.npz")
        datasets.append(
            tuple(
                torch.tensor(a, device=device)
                for a in (
                    images(data["rgb"], data["depth"]),
                    data["proprio"],
                    data["action"] / LIMITS,
                    data["found"].astype(np.float32),
                )
            )
        )
    model = Network().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    train, valid = datasets
    best = float("inf")
    history = []
    for epoch in range(100):
        model.train()
        order = torch.randperm(len(train[0]), device=device)
        for ix in order.split(64):
            prediction = model(train[0][ix], train[1][ix])
            loss = (
                (prediction[:, :4] - train[2][ix]) ** 2
            ).mean() + 0.15 * torch.nn.functional.binary_cross_entropy_with_logits(
                prediction[:, 4], train[3][ix]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            p = torch.cat(
                [
                    model(valid[0][i : i + 64], valid[1][i : i + 64])
                    for i in range(0, len(valid[0]), 64)
                ]
            )
            mse = ((p[:, :4] - valid[2]) ** 2).mean()
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                p[:, 4], valid[3]
            )
            score = float(mse + 0.15 * bce)
        history.append(
            dict(
                epoch=epoch,
                validation_mse=float(mse),
                validation_bce=float(bce),
                score=score,
            )
        )
        if score < best:
            best = score
            torch.save(
                dict(
                    contract=CONTRACT,
                    model={k: v.cpu() for k, v in model.state_dict().items()},
                    epoch=epoch,
                ),
                args.output / "actor.pt",
            )
        if epoch % 10 == 0:
            print(json.dumps(history[-1]), flush=True)
    report = dict(
        history=history,
        best_validation_score=best,
        parameters=sum(p.numel() for p in model.parameters()),
        actor_sha256=hashlib.sha256(
            (args.output / "actor.pt").read_bytes()
        ).hexdigest(),
        dataset_sha256={
            s: hashlib.sha256((args.data / f"{s}.npz").read_bytes()).hexdigest()
            for s in ("train", "validation")
        },
    )
    (args.output / "training.json").write_text(json.dumps(report, indent=2))
    print("Frozen actor ready for held-out closed-loop evaluation", flush=True)


if __name__ == "__main__":
    main()
