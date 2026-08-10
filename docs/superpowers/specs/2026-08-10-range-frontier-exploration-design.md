# Range-Frontier Exploration v2 Design

## Status

Approved architecture, written for implementation review on 2026-08-10. This
spec defines a new single-drone lane. It does not modify or promote the failed
camera-only coverage student, and it grants no shadow, deployment, or flight
authority by itself.

## Goal

Build a recognizable indoor exploration behavior for one Crazyflie 2.1
Brushless with Flow Deck V2, Z-Ranger, and Multi-ranger:

1. build an online occupancy map from real range and Flow telemetry;
2. expose all currently reachable exploration frontiers without prescribing a
   target;
3. use a map-memory PPO policy to choose its own exploration direction and
   speed while reacting to local obstacles;
4. train quickly in a vectorized 2-D simulator, validate in held-out layouts
   and MuJoCo, then evaluate the policy in a bounded live shadow run before it
   can command a flight.

The visible behavior should be exploration, not hovering or a fixed patrol:
advance through open space, slow or turn near an obstacle, select another
frontier, and continue until the time budget or safety monitor ends the run.

## Non-goals

- No semantic navigation, object detection, or AI Deck input in v2.
- No learned altitude, lateral velocity, motor, attitude, or takeoff control.
- No swarm implementation in v2. Interfaces may carry a future `agent_id`, but
  there is no multi-agent observation, reward, or coordination code yet.
- No claim that Flow odometry creates a durable shared world frame. A later
  swarm lane requires Lighthouse, Loco, or another independently verified
  shared localization system.
- No custom radar integration in this slice.

## Governing live evidence

The implementation must preserve the two current passing calibration flights:

- `patrol_1786366257073064000`: 20 Hz device cadence, 1,439 projected range
  points, four active horizontal sensors, 0.723 m trajectory span, 23.61 degree
  yaw span, no speed glitches, minimum battery 3.737 V.
- `patrol_1786366560841486000`: 20 Hz device cadence, 1,488 projected range
  points, four active horizontal sensors, 0.756 m trajectory span, 23.20 degree
  yaw span, no speed glitches, minimum battery 3.653 V.

These runs establish telemetry transport, range projection, and short powered
motion only. They do not establish map accuracy, obstacle-avoidance performance,
or policy authority. Ground reflections previously made `range.front` appear
near while landed; horizontal clearance is therefore interpreted only after the
existing altitude gate confirms flight.

## Architecture

The lane has four isolated units:

1. **Range occupancy mapper** consumes timestamped pose and horizontal ranger
   samples. It owns the persistent map and exposes a body-aligned local crop.
2. **Frontier extractor** consumes only that estimated map and estimated pose.
   It marks every reachable frontier cell but does not select a target.
3. **Range exploration actor** consumes the map with all frontier candidates,
   raw normalized ranger observations, and previous applied action. It chooses
   forward velocity and yaw rate directly, including which frontier or open
   direction to explore.
4. **Safety/runtime shell** owns takeoff, altitude, action limits, stale-data
   handling, clearance overrides, landing, event logs, and bounded cleanup. The
   policy cannot bypass it.

Simulator truth may compute reward and evaluation metrics. It must never enter
the mapper, frontier extractor, actor observation, or live
runtime.

## Versioned observation and action contract

The contract ID is `range-frontier-exploration-v2`.

Actor observation is a flat `float32` vector with these exact segments:

| Segment | Shape | Meaning |
| --- | ---: | --- |
| exploration map | `4 x 32 x 32` | visited, estimated free, occupied, and all reachable frontier cells in a body-aligned 6.4 m square crop |
| horizontal ranges | `4` | front, back, left, right clipped to 4 m and divided by 4 m |
| range validity | `4` | one for a finite return, zero for no-return or stale input |
| previous applied action | `2` | normalized forward and yaw commands after the safety shell |

Total observation size is 4,106. Temporal context is explicit in the persistent
occupancy map and previous applied action; there is no carried neural hidden
state.

The normalized action is `[forward, yaw]`, with `forward` in `[0, 1]` and `yaw`
in `[-1, 1]`. The simulation envelope maps this to 0--0.50 m/s and
-90--90 degree/s. `vy` and `vz` are structurally zero; altitude remains owned by
Crazyflie firmware and the high-level runtime. Live promotion uses stricter
stage-specific caps without changing the actor ABI.

`range.up`, `range.zrange`, battery, Flow quality, roll, pitch, supervisor state,
and telemetry age are safety inputs, not actor inputs.

## Online occupancy mapper

Use a 0.10 m cell sparse grid keyed in the takeoff-relative Flow frame. Each
finite horizontal ray subtracts 0.4 log-odds from traversed cells and adds 0.85
to its endpoint. Values clamp to `[-2.0, 3.0]`; cells at or below `-0.8` are
free, cells at or above `1.7` are occupied, and the rest are unknown. A
no-return sample clears cells out to the 4 m sensor limit but adds no occupied
endpoint. Stale or invalid samples do not update the map.

The mapper uses the existing device timestamp as canonical time and the same
body-to-world convention as `ranger_projection.py`. A discontinuity, decreasing
timestamp, non-finite pose, or arming epoch change resets the map and previous
action rather than attempting recovery through a fallback frame.

The exploration crop contains visited, estimated free, occupied, and reachable
frontier channels. Unknown is represented by zeros in the free and occupied
channels. The 32 by 32 crop samples the persistent 0.10 m map at 0.20 m per
output cell and is rotated into the current body frame, so forward is always up.

## Frontier extraction and baseline

A frontier is an estimated-free cell adjacent to at least one unknown cell.
Frontier cells are grouped with 8-connectivity; groups smaller than three cells
are discarded. The selector flood-fills reachable free cells from the estimated
current cell and rejects groups with no reachable member. Every remaining cell
is exposed in the frontier channel. No target vector, selected group, target
bearing, or planner action is provided to the actor.

For evaluation only, a deterministic classical baseline selects among those
groups. Each group receives:

Each remaining group receives:

`score = unknown_neighbors / (path_length_m + 0.5) - 0.1 * abs(bearing_rad)`

The previous group receives a 10% persistence bonus while it remains reachable,
preventing rapid target oscillation. The selected target is the reachable cell
in the best group closest to its centroid. This target drives only the
non-learned comparison controller. When no frontier is available, the learned
actor still owns the choice to scan, move, or wait; the safety shell does not
inject a preferred turn direction.

## Fast training environment

Implement a Gymnasium-valid single environment and a batched NumPy stepping
core. The world is a connected 2-D rectilinear floor plan with walls, rectangular
obstacles, corridors, and open rooms. The drone is a 0.15 m radius disc at a
fixed 0.40 m flight altitude. Four horizontal sensors use the same body
bearings, 0.03--4.0 m range, no-return semantics, mapper, and frontier extractor
as live runtime. The step rate is 20 Hz.

Training randomizes connected layouts, initial free poses, ranger bias up to
0.05 m, dropout bursts of 0/1/3 frames, odometry scale in `[0.85, 1.15]`, yaw
drift up to 2 degree/s, and action lag of 0/100/250 ms. These are explicit
stress envelopes, not claims of fitted sensor probabilities. The two live
calibration logs fix cadence and command/telemetry conventions; future
ground-truth calibration may narrow the stress envelope but cannot silently
change the contract.

The environment must pass Gymnasium's environment checker and deterministic
seeding tests. The batched core must match the single environment for identical
seeds and actions. MuJoCo remains a slower transfer/challenge backend, not the
primary rollout engine.

## Policy and PPO

Use a small PyTorch map-memory actor-critic: a convolutional encoder for the
four map channels, an MLP for the 10 scalar values, a 64-unit gated per-frame
encoder with a fresh zero state, and separate actor and critic heads. It exposes
no recurrent runtime state and contains 39,131 parameters. PPO uses a
tanh-squashed continuous distribution. Checkpoints bind the exact contract,
source revision,
training configuration, procedural train/selection seeds, state digest, and
evaluation reports. A raw state dict is never promotable.

Two training-only auxiliary losses improve credit assignment without changing
the live observation or action contract. A 0.10-weight consistency loss teaches
the raw policy to agree with forward-motion safety vetoes. During only the
varied close-obstacle curriculum, a 0.10 turn-commitment loss preserves the yaw
sign already chosen by the policy until clearance. It never selects left or
right. The runtime frontier-direction auxiliary weight is zero; PPO owns the
closed-loop direction.

Reward per step is `0.35 * delta_visited_fraction + 0.65 *
delta_observed_free_fraction`, both measured from privileged simulator truth.
This rewards physical exploration as well as sensing and leaves target choice
entirely to the policy. Collision or leaving the valid floor incurs -2.0 and
terminates. Since total possible positive coverage reward is at most 1.0,
collision is always worse than completing the episode. There is no selected
frontier reward and no per-step stall penalty that could prescribe a direction
or make early collision preferable.

## Offline evaluation and promotion gates

A candidate remains simulation-only unless all checks pass on held-out layouts:

- coverage-time AUC and final coverage beat both stationary-scan and classical
  reactive frontier baselines;
- collision rate is zero across the fixed selection suite and no episode leaves
  the valid floor;
- four dedicated, held-out horizontal-range obstacle approaches all begin near
  0.65 m, escape by at least 0.50 m of motion, and have zero collisions or
  safety terminations;
- masking either live ranges or the estimated map materially reduces held-out
  coverage, while the clean policy remains best;
- matched states with the same map, frontier, history, and odometry but different
  front obstruction require different actions and are classified correctly;
- mirrored-frontier response remains a diagnostic, not a promotion gate: a
  policy may consistently choose either valid exploration direction;
- the full range-bias, dropout, odometry-drift, and action-lag stress suite has
  zero collisions and zero safety terminations.

Evaluation reports must explicitly set training, shadow, deployment, and flight
authority independently. Passing simulation plus replay gates permits only a
non-actuating live shadow run.

## Live shadow and first learned-flight boundary

The first hardware use is a normal instrumented scripted patrol with the new
mapper, frontier extractor, actor, and shield running in shadow. The runtime
logs raw telemetry, map updates, all frontier candidates, raw policy actions,
shielded actions, and the actually executed scripted actions at 20 Hz. It never
sends the policy action during shadow.

Shadow fails on telemetry age above 0.25 s, invalid deck/TOC, battery or power
state failure, stale ranger input, Flow quality failure, map reset, non-finite
policy output, clearance override, cleanup timeout, or missing provenance.

Only a passing shadow artifact can make a separately authorized first learned
flight eligible. That flight is limited to 20 s at 0.40 m altitude, 0.20 m/s
forward, and 30 degree/s yaw. It requires the existing manual confirmation,
fresh deck/battery/supervisor checks, a clear indoor test area, an observer, and
bounded land/disarm cleanup. Any horizontal range below 0.35 m suppresses
forward motion; below 0.20 m, stale telemetry, low Flow quality, excessive tilt,
or low-power state aborts to controlled landing. Higher speed stages require a
new passing artifact and explicit user authorization.

## Artifact boundary

Every train, evaluation, shadow, and live run uses a unique directory and an
immutable manifest with input hashes. Derived maps and simulator reports are
evaluation evidence, not training-authoritative live data unless a later data
contract explicitly promotes them. The AI Deck camera artifacts remain separate
and are not relabeled as range-policy evidence.

## Future extension

After one-drone exploration passes, a swarm lane may share per-drone maps and
allocate frontiers centrally while keeping the same local actor ABI. Shared-map
claims require independently verified common-frame localization. Lighthouse is
the first recommended indoor extension; radar or a larger FPV/PX4 platform is a
separate hardware decision driven by outdoor operation, dense 3-D perception,
payload, speed, or endurance requirements.
