"""Evaluate the frozen direct controller; keep all outcomes and one preselected replay."""

import hashlib
import json
import time
from pathlib import Path
from flightrl import _binding
from flightrl.fleet.flight_policy.model import Policy
from flightrl.fleet.flight_policy.course import run

root = Path("artifacts/direct-flight-20260906")
plan = json.loads((root / "plan.json").read_text())
policy = Policy(root / "controller.pt")


# Evaluation must never silently call the classical teacher.
def forbidden(*args):
    raise AssertionError(
        "Classical setpoint controller called during learned evaluation"
    )


_binding.sixdof_setpoint_actions = forbidden
results = {}
started = time.perf_counter()
steps = 0
for seed in plan["test_seeds"]:
    replay = run(policy, seed)
    results[str(seed)] = replay["result"]
    steps += len(replay["records"]) - 1
    if seed == plan["demo_seed"]:
        (root / "replay.json").write_text(json.dumps(replay))
    print(seed, replay["result"], flush=True)
elapsed = time.perf_counter() - started
summary = {
    "results": results,
    "wall_seconds": elapsed,
    "joint_control_steps_per_second": steps / elapsed,
    "aircraft_physics_substeps_per_second": steps * 3 * 5 / elapsed,
    "scope": "CPU learned inference + native physics + collision + recording, no rendering or gradient updates. Three parallel copies; seeds vary course altitudes, not obstacle layouts.",
    "checkpoint_sha256": hashlib.sha256(
        (root / "controller.pt").read_bytes()
    ).hexdigest(),
}
(root / "evaluation.json").write_text(json.dumps(summary, indent=2))
replay = json.loads((root / "replay.json").read_text())
complete = sum(v["status"] == "complete" for v in results.values())
replay["provenance"]["evaluation"] = (
    f"{complete}/{len(results)} altitude variants complete · learned thrust/body rates · prescribed waypoints"
)
replay["provenance"]["checkpoint_sha256"] = summary["checkpoint_sha256"]
Path("viewer/public/fleet/direct-flight.json").write_text(json.dumps(replay))
(root / "source-hashes.json").write_text(
    json.dumps(
        {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path("src/flightrl/fleet/flight_policy").glob("*.py")
        },
        indent=2,
    )
)
print({k: v for k, v in summary.items() if k != "results"})
