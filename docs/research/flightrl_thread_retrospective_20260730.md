# FlightRL thread retrospective

Date: 2026-07-30

This document is the decision-grade retrospective for the camera-conditioned
Crazyflie work. It synthesizes the longer experiment ledger, the research
review, simulation reports, and live-flight summaries. It does not approve
unrestricted learned flight or raw motor authority.

## Executive verdict

The project did make material progress, but three different achievements were
often discussed as if they were the same:

1. The AI Deck camera and transport path now work reliably enough for policy
   input. The stable contract is `64x48` grayscale at about 65 FPS with complete
   frames and concurrent telemetry.
2. A Puffer policy has controlled a real drone in a bounded waypoint flight,
   but only as a lateral residual. Deterministic logic supplied the forward
   waypoint. This is a real learned-control demo, not semantic navigation.
3. Fixed-door student v59 is the first camera-causal recurrent policy with
   useful semantic mission behavior in simulation. It reaches 79.11% success,
   74.73% outside-FOV success, and 0.56% collision. It has not yet controlled a
   real flight.

The shortest path to an honest filmable semantic demo is therefore not another
large training run. It is:

1. Record a hash-bound v59 shadow trace on the real AI Deck camera.
2. If direction, freshness, timing, and finite-action checks pass, give v59 yaw
   authority up to 8 deg/s while firmware holds position and height.
3. Film the drone finding and centering a door that begins 30-60 degrees outside
   the camera FOV.
4. Add a short bounded forward approach only after the yaw-only flight passes.

That demo would show the actual recurrent student controlling the drone from
camera evidence. It would not yet claim room-scale exploration, obstacle
avoidance, multi-object language conditioning, or on-edge inference.

## What we have actually demonstrated

| Surface | Strongest evidence | What it proves | What it does not prove |
| --- | --- | --- | --- |
| AI Deck | Stable `64x48` gray4 stream near 64.86 FPS, zero drops, concurrent radio | Camera observations can reach the policy loop at useful rate | Semantic perception or control |
| Native visual sim | `fast16_lowlight_v5`: 98.59% obstacle success, 1.33% collision; masked camera 0% success and 100% collision | Low-resolution camera-causal local visual control is learnable in the native lane | Object discovery or language missions |
| Real waypoint | 0.70 m flights at 0.08, 0.16, and 0.32 m/s reached target with learned lateral residual active | Puffer inference, AI Deck frames, telemetry, firmware stabilization, and bounded learned authority work together | Policy-selected forward motion or semantic navigation |
| Fixed-door teacher | 93.55% success, 90.68% outside-FOV success, 0.22% collision | The task and privileged label generator are mostly feasible | A deployable policy |
| Fixed-door v59 | 79.11% success, 74.73% outside-FOV success, 0.56% collision; masked success 1.53% | A recurrent deployable student can search for and approach the fixed category from pixels and nonprivileged state in simulation | Real transfer, obstacle-rich safety, desk/monitor commands, or a shared semantic policy |

The corrected v59 reevaluation is authoritative. The original report's masked
camera result retained stale detector evidence and was invalid. The corrected
ablation masks both pixels and camera-derived detector evidence.

## Chronology and decisions

### 1. Deck-independent control and early transfer

The broken Ranger Deck forced the system toward modular observations and
firmware-stabilized control. This was useful. It prevented the learned policy
from depending on one fragile sensor and established that firmware should
retain arming, takeoff, attitude, altitude, landing, and abort behavior.

The costly mistake was trying direct raw control before command signs,
previous-action scaling, and real drift distributions matched simulation. A
high-authority live run drifted/crashed, and analysis found a pitch-sign
mismatch plus weak recovery-state coverage. That result correctly moved live
work back to bounded setpoints and residual authority.

### 2. AI Deck recovery and vision contract

The AI Deck initially appeared unavailable because the GAP8 bootloader needed
restoration. JTAG recovery fixed that hardware blocker. The next blocker was
not perception but frame integrity: some decoded images contained horizontal
black regions or two raster-wrapped room views. Triple-buffered complete-frame
transport and whole-frame dropping under backpressure fixed the latest image.

The stable policy-oriented camera contract is current grayscale plus temporal
information. Frame difference is useful as an auxiliary channel, but it cannot
replace the current frame because a stationary door would disappear. Propeller
guard regions are deterministic self-occlusion and are masked consistently.

The camera work produced real systems value:

- about 64.86 FPS at `64x48` gray4 with zero drops;
- about 111 FPS in a lower-fidelity stress configuration;
- measured camera, preprocessing, transport, inference, and telemetry timing;
- a tunable observation boundary suitable for offboard and later GAP8 inference.

The 65 FPS mode remains the stable baseline. The 100+ FPS mode is a throughput
experiment, not a validated policy input.

### 3. Native local visual control

The native C environment was the strongest engineering direction. It combines
six-DoF physics, procedural scenes, simple rendering, fused vector stepping,
and Puffer collection without a Python environment loop. Measured throughput
was about 2.05M environment transitions/s, about 81k learning transitions/s,
and 220-292k transitions/s in maximum-throughput profiles.

The successful ingredients were:

- recurrent expert bootstrap;
- episode-level dynamics, layout, lighting, texture, and sensor randomization;
- low-resolution current and temporal camera channels;
- masked-camera causal evaluation;
- retaining the best bootstrap when PPO regressed it.

Raw PPO, reward reshaping, entropy changes, and simply adding transitions were
not reliable. A 262k-transition candidate outperformed a longer continuation.
This established a rule: training duration is a controlled variable, not a
monotonic quality knob.

### 4. Real bounded waypoint flights

The live waypoint sequence validated the hardware/control contract:

| Run | Learned authority | Outcome | Interpretation |
| --- | --- | --- | --- |
| `visual_waypoint_v5_live2` | none reached | warm-up abort | No policy-control evidence |
| `visual_waypoint_v5_live3` | lateral velocity only | safety abort at 0.186 m cross-track | Safety gate worked; residual was active |
| deterministic 0.35 m | none | target reached | Firmware/Flow baseline |
| Puffer 0.70 m, 0.08 m/s | lateral velocity only | target reached | First complete bounded Puffer-in-loop flight |
| Puffer 0.70 m, 0.16 m/s | lateral velocity only | target reached | Faster contract remained stable |
| Puffer 0.70 m, 0.32 m/s | lateral velocity only | target reached | Faster deterministic path remained stable; residual was very small |

These repetitions were initially useful because they isolated propeller,
battery, Flow Deck, command, drift, and speed issues. Once three 0.70 m runs
passed, further deterministic-forward repeats stopped answering the semantic
question. This is where the project should have switched earlier to a
student-selected action gate.

### 5. Broad MuJoCo semantic stack

The Python/MuJoCo semantic lane tried to solve grounding, mapping, recurrent
exploration, obstacle safety, approach, and multi-target missions together.
It achieved useful partial evidence, but became the main experimental loop
that stopped producing clear information.

Representative outcomes:

- v5: 25% completion, 75% discovery, 0% collision;
- v7 with spatial memory: 60% completion on a small four-room screen;
- v9 with 16 rooms at the same budget: 0% completion, 25% discovery;
- v11 with a larger budget and 12 rooms: 38.89% completion, 5.56% collision;
- v24-v28: roughly 6-24% completion with safety/completion tradeoffs.

The full lane ran at about 270 SPS. More importantly, each version often changed
several variables: room distribution, labels, memory, safety loss, reward, and
selection metric. The result was metric motion without causal attribution.
Short two-room screens also promoted candidates that failed the full held-out
evaluation.

MuJoCo itself was not the problem. The problem was using a slow, integrated
Python loop as the primary bulk-training lane. MuJoCo remains valuable as an
independent contact/dynamics validator.

### 6. Grounding and room diversity

Room diversity did help when the bottleneck was isolated correctly. A compact
door grounder trained on 24 rooms overfit: training AUROC was 0.9644 while
held-out AUROC was 0.7924. Keeping model capacity fixed and increasing to 128
training rooms improved held-out AUROC to 0.9127 and centroid error to 0.0912.

That does not mean every policy improves with more rooms. In the integrated
control lane, adding rooms without increasing samples per factor hurt. The
right unit is a factorized scene distribution, not a nominal room count:
topology, target geometry, materials, light, occlusion, distractors, start pose,
camera parameters, latency, and dynamics must be resampled and evaluated on
disjoint combinations.

The native-to-real grounder work also revealed a renderer gap. Some heads passed
native AUROC but failed real false-positive or centroid gates. Reusing the same
small real manifest for repeated model selection made it development data, not
a final holdout. This gate prevented a strong synthetic number from becoming a
false live-readiness claim.

### 7. Fixed-door reset

The research review recommended one fixed-category task before shared language
conditioning: find an interior door, including when it starts outside FOV,
approach it, and stop at 0.80 m. Privileged geometry is available only to the
teacher and training critic. The deployable recurrent actor receives camera
channels, camera-derived evidence, telemetry, phase, previous executed action,
and recurrent state.

Versions v45-v58 exposed the dominant training failure:

- increasing DAgger fraction did not unlock the old controller;
- stock and asymmetric PPO improved some scores but caused high collision;
- observation-matched teachers helped only slightly;
- a detector-matched checkpoint improved a screen metric while collision rose
  to 41.51%;
- reusing the old control initialization kept the actor in an inert or unsafe
  behavior regime.

Version v59 retained the perception tensors but reinitialized fusion,
recurrence, and action decoding. Pure behavioral cloning over 1,048,576 samples
then reached 79.11% success and 0.56% collision. This was the decisive
experiment. It showed that the previous local minimum was architectural
initialization, not insufficient rooms, PPO steps, or reward tuning.

## What the research predicted correctly

The current evidence agrees with the strongest external results:

- privileged teacher to deployable visual student is a good decomposition;
- temporal state is important for outside-FOV search and intermittent evidence;
- iterative on-policy imitation is the correct next way to cover student
  failures, but only after the student architecture can express useful control;
- asymmetric critics may help fine-tune long-horizon behavior, but PPO should
  not be assumed to improve imitation;
- tiny grayscale policies can run fast enough for navigation above firmware
  stabilization;
- explicit randomized simulation should precede a learned world model;
- firmware-stabilized setpoints are the defensible near-term authority boundary.

FlightRL now has local evidence for the first, second, fifth, sixth, and seventh
points. It has not yet run the decisive fresh-controller DAgger comparison.

## Where our approach diverged from the evidence

1. We changed multiple variables sequentially on one seed instead of running a
   fixed, equal-budget comparison.
2. We expanded to multi-component semantic navigation before proving one fixed
   category.
3. We treated more rooms, more transitions, and PPO as generic improvements.
4. We used the slow MuJoCo/Python lane for bulk iteration instead of native C.
5. We repeatedly optimized proxy gates after they stopped predicting mission
   control.
6. We did not freeze the live action scale before v59 training.
7. The current v59 training override disabled obstacles, so its low collision
   rate is not evidence of clutter avoidance.

The research-backed gate was still useful. It caught camera leakage, renderer
gap, stale evidence, unsafe PPO candidates, and the difference between a teacher
and deployable student. The mistake would be using an arbitrary threshold to
delay a bounded demo after the risk-relevant checks pass. v59 misses the 80%
completion target by 0.89 percentage points but passes the outside-FOV,
collision, and corrected camera-causality checks.

## Perspectives that change the next decision

### Control and safety

The first semantic live action should be yaw only. Translation adds optical-flow
drift, standoff estimation, and collision consequences without being necessary
to prove visual search. Firmware must retain takeoff, position/height hold,
landing, and abort. Stale frames or actions command zero yaw.

### Learning

Do not discard the teacher because it is good. The teacher sees privileged
state and cannot run on the real drone. Its purpose is to label what the
deployable student should do, especially on failure states. v59 proves the
student can imitate useful behavior. Fresh-controller DAgger is now the highest
value training experiment because earlier DAgger failures used the poisoned old
controller and do not answer this question.

### Perception and sim-to-real

The door-specific checkpoint is acceptable for the first demo. It tests active
visual search without requiring a language encoder. It cannot find a desk or
monitor unless retrained. A later shared policy must receive a target token or
compact grounded target representation and pass same-scene token swaps.

### Systems

Puffer/native C is the bulk-training lane. MuJoCo and a richer renderer should
challenge physics and appearance independently. Environment SPS, learner SPS,
camera FPS, and inference latency must remain separate metrics.

### Experimental method

Every new run needs one written hypothesis, one primary metric, a frozen
baseline, equal budget, and a stop condition. A new version number is not an
experiment. Three seeds are the practical minimum for the next comparison;
five remain the research target.

### Demo and product

The first strong claim should be narrow and visually legible: "A recurrent
camera policy searches for and faces an interior door while the Crazyflie
firmware stabilizes flight." Full autonomous room exploration is a later claim
because obstacle-rich training and real transfer are not yet established.

## Continuation

The exact artifact, live sequence, v60 experiment, stop rules, and milestone
definition are in `docs/research/flightrl_continuation_handoff_20260730.md`.
The chronological quality audit and frozen experiment protocol are in
`docs/research/flightrl_experiment_control_protocol_20260730.md`.

## Primary references

- `docs/research/flightrl_experiment_retrospective_20260729.md`
- `docs/research/state_of_the_art_architecture_review_20260729.md`
- `docs/research/fixed_door_observability_20260729.md`
- `docs/native_visual_training_fast_lane.md`
- `docs/research/vision_observation_contract_20260724.md`
- `docs/research/flightrl_continuation_handoff_20260730.md`
- `docs/research/flightrl_experiment_control_protocol_20260730.md`
