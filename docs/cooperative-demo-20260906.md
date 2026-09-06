# Cooperative inspection takeover — 6 September 2026

A local simulation demo is available at `/fleet.html`, under **Cooperative takeover**.
Three FPV-sized drones divide nine waypoint-inspection targets. At 8 seconds,
Drone 2 becomes unavailable and holds position; after a 200 ms notification delay,
the coordinator releases its unfinished task. A teammate takes it over. Remaining
active drones complete the targets and return home. The predetermined demonstration
layout (test seed 120) completes in **45.5 simulated seconds** with no collision.

## What was trained

A shared **4,929-parameter neural route-cost bidder**, trained by supervised learning
on 8,000 A-star route labels from forest seeds 20–35. Validation used 800 pairs from
seeds 50–53. Training ran 100 epochs; the checkpoint with the lowest validation mean
absolute error was frozen before testing. Validation error: **0.209 m**.

The model sees positions, target coordinates and a summary of known obstacles. Its
predicted bid selects the next available task for each idle aircraft. It does not
consume RGB-D images or issue low-level flight commands. Ownership, failure handling,
A-star routing, altitude lanes and the native six-DOF setpoint controller are explicit
algorithms. Assignment is centrally coordinated, not an emergent decentralized policy.

## Held-out evaluation

Test seeds 120–131 were not used for training or checkpoint selection. Each mission
requires every target to be dwell-inspected once, no swept environment/peer collision,
and all active drones back at their launch positions. The camera renderer is excluded
from the simulation experiment; the dashboard re-renders the recorded poses.

| Controller / task allocation | Complete missions | Mean simulated duration |
| --- | ---: | ---: |
| Learned bids, dynamic takeover, drone fault | 12 / 12 | 54.08 s |
| Learned bids, no drone fault | 12 / 12 | 46.43 s |
| Learned bids, fixed task ownership, drone fault | 0 / 12 | 239.9 s timeout |
| Exact route-cost bids, dynamic takeover, drone fault | 12 / 12 | 53.70 s |
| Nearest-target bids, dynamic takeover, drone fault | 12 / 12 | 53.98 s |

The learned fault arm's minimum measured peer surface separation was **0.509 m**.
The baseline comparison establishes that reassignment solves stranded work. It does
**not** establish an advantage for learned bids over simpler dynamic allocation.
A failure can change assignment order enough to shorten some individual runs; this
is not evidence that failures help in general. This is one bounded forest family,
with ideal simulator pose telemetry and explicit altitude separation.

## Limits

Inspection is a one-second arrival/dwell surrogate, not visual defect recognition.
Unavailable means a controlled hold, not motor failure or a falling aircraft.
Motor responses remain surrogate dynamics. No aerodynamic coupling, battery model,
real communication transport, detailed mesh collision or camera-conditioned policy
was validated. No new cross-drone or cross-environment transfer claim is made.

## Evidence and reproduction

- `artifacts/cooperative-demo-20260906/plan.json`: frozen splits, demo seed and limits.
- `training.json`, `bids.pt`: actual trained checkpoint and validation metric.
- `results.json`: all 60 evaluation runs, including failures and baselines.
- `replay.json`: the predetermined held-out run, event times, task ownership and poses.
- `source-hashes.json`: source identity.
- Train: `PYTHONPATH=src .venv/bin/python scripts/train_cooperative_demo.py` (requires a new output directory).
- Package: `PYTHONPATH=src .venv/bin/python scripts/package_cooperative_review.py`.

Five focused tests cover conservative routes, unique task ownership, delayed failure
release, takeover completion, the stranded fixed-assignment baseline and peer-link
contracts. Viewer TypeScript and production build pass.
