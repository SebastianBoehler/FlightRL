"""Record predetermined reusable mission demos and retain every evaluation outcome."""

import hashlib
import json
from pathlib import Path
from flightrl.fleet.cooperative.bids import LearnedBids
from flightrl.fleet.cooperative.mission import run

root = Path("artifacts/mission-catalog-20260906")
root.mkdir(exist_ok=False)
checkpoint = Path("artifacts/cooperative-demo-20260906/bids.pt")
bid = LearnedBids(checkpoint)
plan = {
    "seeds": list(range(142, 148)),
    "demo_seed": 142,
    "development_seeds": [140, 141],
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "new_training": False,
    "missions": ["search-rescue", "industrial"],
}
(root / "plan.json").write_text(json.dumps(plan, indent=2))
results = {}
for mission, family, kind in [
    ("search-rescue", "forest", "search_rescue"),
    ("industrial", "utility_plant", "inspection"),
]:
    for seed in plan["seeds"]:
        replay = run(seed, bid=bid, mission=kind, family=family, failure_s=None)
        results[f"{mission}/{seed}"] = replay["result"]
        if seed == plan["demo_seed"]:
            replay["provenance"]["checkpoint_sha256"] = plan["checkpoint_sha256"]
            (root / f"{mission}.json").write_text(json.dumps(replay))
        print(
            mission,
            seed,
            replay["result"]["status"],
            replay["result"]["mission_time_s"],
            flush=True,
        )
    selected = [v for k, v in results.items() if k.startswith(mission + "/")]
    complete = sum(v["status"] == "complete" for v in selected)
    path = root / f"{mission}.json"
    replay = json.loads(path.read_text())
    replay["provenance"]["evaluation"] = (
        f"{complete}/{len(selected)} evaluation layouts complete · learned task bids + explicit XYZ navigation"
    )
    (Path("viewer/public/fleet") / f"{mission}.json").write_text(json.dumps(replay))
(root / "results.json").write_text(json.dumps(results, indent=2))
(root / "source-hashes.json").write_text(
    json.dumps(
        {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path("src/flightrl/fleet/cooperative").glob("*.py")
        },
        indent=2,
    )
)
