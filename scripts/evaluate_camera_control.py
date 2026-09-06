"""Frozen evaluation and sensor ablations; never invoke a privileged controller."""

import hashlib
import json
import time
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from flightrl import _binding
from flightrl.fleet.camera_policy.network import Policy
from flightrl.fleet.camera_policy.episode import run
from flightrl.fleet.camera_policy.contract import contract

parser = argparse.ArgumentParser()
parser.add_argument("artifact_dir", type=Path)
root = parser.parse_args().artifact_dir
plan = json.loads((root / "plan.json").read_text())
if (root / "evaluation.json").exists():
    raise FileExistsError("Evaluation already recorded")
policy = Policy(root / "actor.pt")


def forbidden(*args):
    raise AssertionError("Teacher called during camera actor evaluation")


_binding.sixdof_setpoint_actions = forbidden
results = {}
started = time.perf_counter()
camera_steps = 0
for mode in ["normal", "no_images", "no_messages"]:
    for seed in plan["test_seeds"]:
        demo = mode == "normal" and seed == plan["demo_seed"]
        replay, images = run(
            seed, policy, None if mode == "normal" else mode, save_images=demo
        )
        results[f"{mode}/{seed}"] = replay["result"]
        camera_steps += len(replay["records"]) * 3
        if demo:
            atlas = (
                np.stack(images)
                .transpose(0, 2, 1, 3, 4)
                .reshape(len(images) * 48, 3 * 64, 3)
            )
            Image.fromarray(atlas).save(root / "sensor-rgb.png")
            replay["sensor_atlas"] = "/fleet/camera-control-rgb.png"
            (root / "replay.json").write_text(json.dumps(replay))
        print(
            mode,
            seed,
            replay["result"]["status"],
            replay["result"]["reports"],
            flush=True,
        )
summary = {
    mode: sum(
        v["status"] == "complete"
        for k, v in results.items()
        if k.startswith(mode + "/")
    )
    for mode in ["normal", "no_images", "no_messages"]
}
elapsed = time.perf_counter() - started
report = {
    "results": results,
    "complete_counts": summary,
    "cases_per_mode": len(plan["test_seeds"]),
    "wall_seconds": elapsed,
    "agent_camera_control_steps_per_second": camera_steps / elapsed,
    "throughput_scope": "CPU native RGB-D rendering, neural inference, physics, collisions, logging and demo PNG export; no gradient updates",
    "checkpoint_sha256": hashlib.sha256((root / "actor.pt").read_bytes()).hexdigest(),
}
(root / "evaluation.json").write_text(json.dumps(report, indent=2))
(root / "io-contract.json").write_text(json.dumps(contract(), indent=2))
replay = json.loads((root / "replay.json").read_text())
waits = sum(
    v["status"] == "timeout" and v["reports"] == [True, True, False]
    for k, v in results.items()
    if k.startswith("no_messages/")
)
replay["provenance"]["evaluation"] = (
    f"Camera actor: {summary['normal']}/12 complete · messages withheld: {waits}/12 wait · RGB-D removed: {summary['no_images']}/12 complete"
)
replay["provenance"]["checkpoint_sha256"] = report["checkpoint_sha256"]
Path("viewer/public/fleet/camera-control.json").write_text(json.dumps(replay))
Path("viewer/public/fleet/camera-control-rgb.png").write_bytes(
    (root / "sensor-rgb.png").read_bytes()
)
(root / "source-hashes.json").write_text(
    json.dumps(
        {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path("src/flightrl/fleet/camera_policy").glob("*.py")
        },
        indent=2,
    )
)
print(summary, flush=True)
