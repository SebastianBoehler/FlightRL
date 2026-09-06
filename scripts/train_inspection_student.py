"""Local bounded demonstration/corrective distillation, selected on validation only."""

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from flightrl.artifact_identity import sha256_file
from flightrl.inspection.scenarios import SPLITS, GATES, scenario
from flightrl.inspection.rollout import run_mission
from flightrl.inspection.student import VisualController, StudentPolicy, image_tensor


def train(data, seed, path, epochs=35):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = VisualController().to("mps")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    rgb = np.stack([x[0] for x in data])
    depth = np.stack([x[1] for x in data])
    proprio = torch.tensor(np.stack([x[2] for x in data]), device="mps")
    target = torch.tensor(np.stack([x[3] for x in data]), device="mps")
    images = image_tensor(rgb, depth, "mps")
    # Independent short sequences; episode boundaries supplied as reset markers by collection.
    starts = np.array(
        [i for i in range(len(data) - 8) if not any(x[4] for x in data[i + 1 : i + 8])]
    )
    if not len(starts):
        raise ValueError("no valid demonstration sequences")
    for epoch in range(epochs):
        losses = []
        for _ in range(30):
            ix = rng.choice(starts, 64)[:, None] + np.arange(8)[None, :]
            ix = torch.tensor(ix, device="mps")
            prediction, _ = model(images[ix], proprio[ix])
            loss = ((prediction - target[ix]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch % 10 == 0:
            print(f"seed {seed} epoch {epoch}: loss {np.mean(losses):.5f}", flush=True)
    torch.save(
        {"model": {k: v.cpu() for k, v in model.state_dict().items()}, "seed": seed},
        path,
    )
    return {
        "seed": seed,
        "final_loss": float(np.mean(losses)),
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--industrial", action="store_true")
    args = parser.parse_args()
    global scenario, run_mission, GATES
    if args.industrial:
        from flightrl.inspection.industrial import utility_plant
        from functools import partial

        GATES = {**GATES, "mission_ticks": 1800}
        scenario = utility_plant
        run_mission = partial(run_mission, industrial=True)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    start = time.perf_counter()
    (args.output / "evaluation-plan.json").write_text(
        json.dumps({"splits": SPLITS, "gates": GATES}, indent=2)
    )
    data = []
    teachers = []
    for seed in SPLITS["train"]:
        result, _, _, _, samples = run_mission(
            scenario(seed),
            ticks=1800 if args.industrial else 900,
            seed=seed,
            collect=True,
        )
        data.extend((*x, i == 0) for i, x in enumerate(samples))
        teachers.append(result)
    print(f"Collected {len(data)} teacher samples", flush=True)
    results = []
    for seed in (0, 1, 2):
        path = args.output / f"student-{seed}.pt"
        result = train(data, seed, path)
        policy = StudentPolicy(path)
        result["validation"] = [
            run_mission(
                scenario(i),
                ticks=1800 if args.industrial else 900,
                policy=policy,
                seed=i,
            )[0]
            for i in SPLITS["validation"]
        ]
        result["score"] = sum(
            r["coverage"] - float(r["collision"]) for r in result["validation"]
        )
        results.append(result)
        print("validation", seed, result["score"], flush=True)
    selected = max(results, key=lambda r: r["score"])["seed"]
    # Corrective demonstrations on selected student's states, using the same-information teacher.
    path = args.output / f"student-{selected}.pt"
    policy = StudentPolicy(path)
    correction = []
    for seed in SPLITS["train"][:4]:
        *_, samples = run_mission(
            scenario(seed),
            ticks=1800 if args.industrial else 900,
            seed=seed,
            policy=policy,
            collect=True,
        )
        correction.extend((*x, i == 0) for i, x in enumerate(samples))
    corrected = args.output / "student-corrected.pt"
    correction_result = train(data + correction, selected, corrected, epochs=35)
    correction_result["validation"] = [
        run_mission(
            scenario(i),
            ticks=1800 if args.industrial else 900,
            policy=StudentPolicy(corrected),
            seed=i,
        )[0]
        for i in SPLITS["validation"]
    ]
    correction_result["score"] = sum(
        r["coverage"] - float(r["collision"]) for r in correction_result["validation"]
    )
    chosen = (
        corrected if correction_result["score"] >= results[selected]["score"] else path
    )
    (args.output / "selected.pt").write_bytes(chosen.read_bytes())
    report = {
        "environment": "utility_plant_with_gusts_and_dust"
        if args.industrial
        else "inspection_room",
        "method": "recurrent_visual_behavior_cloning_plus_corrective_demonstrations",
        "teacher": "same_information_classical_rgbd",
        "samples": len(data),
        "corrective_samples": len(correction),
        "seeds": results,
        "correction": correction_result,
        "selected": chosen.name,
        "checkpoint_sha256": sha256_file(args.output / "selected.pt"),
        "wall_s": time.perf_counter() - start,
        "device": "Apple MPS",
        "teacher_results": teachers,
        "splits": SPLITS,
        "gates": GATES,
    }
    (args.output / "training.json").write_text(json.dumps(report, indent=2))
    print("Training finished", flush=True)


if __name__ == "__main__":
    main()
