"""Publish the predetermined test seed, never choose a demo based on outcome."""

import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
experiment = root / "artifacts/cooperative-demo-20260906"
plan = json.loads((experiment / "plan.json").read_text())
data = json.loads((experiment / "replay.json").read_text())
results = json.loads((experiment / "results.json").read_text())
assert data["result"]["seed"] == plan["demo_seed"]
held_out = [r for key, r in results.items() if key.startswith("learned_fault/")]
success = sum(r["status"] == "complete" for r in held_out)
data["provenance"]["evaluation"] = (
    f"Learned allocation · {success}/{len(held_out)} held-out failure missions complete · known-map flight"
)
target = root / "viewer/public/fleet/cooperative.json"
target.write_text(json.dumps(data))
print(target)
