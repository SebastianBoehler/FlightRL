# Navigation Autonomy

FlightRL now has a small SOTApilot-inspired navigation layer for single-drone,
range/telemetry-based work. It deliberately does not add camera/depth
perception or multi-agent behavior.

## Benchmark Reports

Build a report from scenario metric records:

```bash
python scripts/build_navigation_benchmark_report.py \
  --input artifacts/replay/navigation_records.json \
  --output artifacts/replay/navigation_benchmark.json
```

Input records use this shape:

```json
{
  "records": [
    {
      "label": "candidate",
      "scenario": "obstacle_room",
      "checkpoint": "artifacts/checkpoints/candidate.pt",
      "metrics": {
        "mean_completed_fraction": 0.95,
        "mean_survival_fraction": 0.98,
        "mean_position_error_m": 0.2,
        "clearance_p01_m": 0.32,
        "action_saturation_fraction": 0.04
      }
    }
  ]
}
```

Default scenarios are defined in `flightrl.navigation.scenarios`:

- `target_approach`
- `obstacle_room`
- `vertical_clearance`
- `recovery`
- `hold_or_land`

## Candidate Bundles

Create a reproducible candidate manifest after benchmark scoring:

```bash
python scripts/build_navigation_candidate_bundle.py \
  --name candidate \
  --checkpoint artifacts/checkpoints/candidate.pt \
  --benchmark artifacts/replay/navigation_benchmark.json \
  --output-dir artifacts/candidates/candidate
```

The bundle records checkpoint provenance, benchmark provenance, schema names,
hardware eligibility, and future extension fields. Passing the navigation
benchmark marks a candidate as `shadow_only`, not live-flight approved.

## Mission State Machine

`flightrl.navigation.mission` defines the high-level single-drone mission flow:

```text
preflight -> takeoff -> search -> navigate -> recover -> hold -> land -> abort
```

The state machine is pure Python and has no hardware side effects. Live runners
can use it later to decide which controller or policy is active while preserving
phase-specific speed/yaw limits.
