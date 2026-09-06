"""Frozen supervised coordination experiment with untouched test layouts."""

import hashlib
import json
from pathlib import Path
from flightrl.fleet.cooperative.bids import train, LearnedBids
from flightrl.fleet.cooperative.mission import run

root = Path("artifacts/cooperative-demo-20260906")
root.mkdir(exist_ok=False)
plan = {
    "train_seeds": list(range(20, 36)),
    "validation_seeds": list(range(50, 54)),
    "test_seeds": list(range(120, 132)),
    "demo_seed": 120,
    "targets": 9,
    "drones": 3,
    "failure_s": 8.0,
    "failure_notification_delay_s": 0.2,
    "learned": "Supervised route-cost bids used to select the next available task",
    "explicit": "Central ownership/reassignment, A-star on known map, altitude lanes, native six-DOF setpoint control",
    "observations": "Simulator pose, known target positions and obstacle summary; no camera inputs",
    "success": "All nine targets dwell-inspected once, active drones return home, no swept collisions",
    "limits": "One forest family and FPV size; no aerodynamic coupling, battery physics, or learned low-level flight; telemetry ideal except declared fault notification delay",
}
(root / "plan.json").write_text(json.dumps(plan, indent=2))
training = train(root / "bids.pt", plan["train_seeds"], plan["validation_seeds"])
training["checkpoint_sha256"] = hashlib.sha256(
    (root / "bids.pt").read_bytes()
).hexdigest()
(root / "training.json").write_text(json.dumps(training, indent=2))
print(training, flush=True)
bid = LearnedBids(root / "bids.pt")
results = {}
for seed in plan["test_seeds"]:
    for arm, failure, mode, cost in [
        ("learned_fault", 8.0, "dynamic", bid),
        ("learned_nominal", None, "dynamic", bid),
        ("fixed_fault", 8.0, "fixed", bid),
        ("oracle_fault", 8.0, "dynamic", None),
        (
            "nearest_fault",
            8.0,
            "dynamic",
            lambda r, a, b: float(((a[:2] - b[:2]) ** 2).sum() ** 0.5),
        ),
    ]:
        replay = run(
            seed,
            bid=cost,
            mode=mode,
            failure_s=failure,
            record=seed == plan["demo_seed"] and arm == "learned_fault",
        )
        results[f"{arm}/{seed}"] = replay["result"]
        print(
            arm,
            seed,
            replay["result"]["status"],
            replay["result"]["mission_time_s"],
            flush=True,
        )
        if seed == plan["demo_seed"] and arm == "learned_fault":
            replay["provenance"]["checkpoint_sha256"] = training["checkpoint_sha256"]
            (root / "replay.json").write_text(json.dumps(replay))
    (root / "results.json").write_text(json.dumps(results, indent=2))
sources = {
    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in Path("src/flightrl/fleet/cooperative").glob("*.py")
}
(root / "source-hashes.json").write_text(json.dumps(sources, indent=2))
