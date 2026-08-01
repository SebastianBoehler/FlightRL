# FlightRL experiment retrospective and research handoff

Date: 2026-07-29. Updated with the fixed-door D1 lineage on 2026-07-30.

Status: architecture review required before further semantic-policy tuning.
No checkpoint listed here is approved for unrestricted live flight.

## Executive verdict

FlightRL has not been uniformly stuck. Two materially different paths have been
mixed together:

1. The native C/Puffer visual-control path is fast and has a strong simulation
   result for local waypoint following and obstacle avoidance.
2. The Python-looped MuJoCo semantic path is slow and has not produced a safe,
   general semantic-search policy.

The strongest current visual checkpoint is
`flightrl_visual_fast16_lowlight_v5_1048576`. It reaches 98.59% success and
1.33% collision in held-out randomized obstacle scenes, while camera masking
reduces success to zero. Its native environment, including rendering, reaches
about 2.05 million transitions/s. It does not identify semantic targets: it
receives a privileged six-value goal intent and solves local visual control.

The best larger semantic run reached 38.89% mission success with 5.56%
collision. Later safety changes reduced some proxy errors but did not remove
the completion/collision tradeoff. All semantic checkpoints failed the shadow
gate. Full semantic training runs execute at roughly 270 steps/s.

The next move is therefore not another `v29` reward or replay tweak. Freeze the
current semantic lineage as evidence, research prior work, then rebuild semantic
navigation as independently gated components on the native training substrate.

## Intended system boundary

The near-term product remains:

```text
mission text or target phrase
  -> semantic grounder or compact target token
  -> recurrent visual exploration/navigation policy
  -> bounded velocity, altitude, and yaw-rate setpoints
  -> stock Crazyflie estimator and stabilizer
  -> motors
```

Host-side grounding and inference are acceptable for development. The design
must preserve a path to a tiny quantized actor on GAP8/STM32. Direct motor
control remains a separate long-term lane and must not be conflated with
semantic navigation validation.

## Evidence inventory

The repository currently contains 308 JSON reports, summaries, or gates. The
two recent camera-policy lineages contain 27 native visual reports and 36
semantic reports. Smoke tests, duplicate live gates, and short probes are useful
debugging evidence but are not treated as comparable checkpoint evaluations.

## Native visual-control lineage

The hot loop contains native six-DoF physics, procedural rooms, ray rendering,
temporal observation assembly, reward, termination, and reset. PufferLib
collects vectorized rollouts; Python configures training and optimizes the small
encoder/MinGRU policy.

| Checkpoint | Steps | Selected | Obstacle success | Collision | Masked-camera result | Verdict |
| --- | ---: | --- | ---: | ---: | --- | --- |
| `fast16_cpu_4194304` | 4.19M | PPO | 0% | 100% | not useful | Learned clear-room prior only |
| `fast16_avoid_8388608` | 8.39M | PPO | 0% | 43.5% | similar failure | Weak visual causality |
| `fast16_entropy_8388608` | 8.39M | PPO | 41.7% | 11.1% | nearly 100% collision | Incomplete |
| `fast16_bounded_8388608` | 8.39M | PPO | 45.32% | 54.52% | 0% success | Causal but unsafe |
| `fast16_bootstrap_v1_16384` | 16K | bootstrap | 99.90% | 0.10% | camera dependent | Narrow early result |
| `fast16_randomized_v1_262144` | 262K | bootstrap | 100% | 0% | 0% success | Strong randomized result |
| `fast16_randomized_v3_262144` | 262K | PPO | 99.92% | 0.08% | 0% success | Simulation gate passed |
| `fast16_randomized_v4_1048576` | 1.05M | candidate | 92.97% | 7.03% | camera dependent | More training regressed |
| `fast16_lowlight_v5_1048576` | 1.05M | bootstrap | 98.59% | 1.33% | 0% success, 100% collision | Current local-control baseline |

The low-light v5 policy has 33,761 parameters. Clear-room success is 99.84%
with zero collision; nominal obstacle success is 98.51% with 1.41% collision.
Its stationary live shadow processed 650 `64x48` gray4 frames at 64.86 FPS with
zero drops and bounded actions. That validates the observation/runtime path,
not autonomous object search or unrestricted flight.

### Native visual lesson

Explicit recurrent expert bootstrap, bounded residual control, causal camera
tests, and domain randomization worked. Raw PPO, entropy/reward adjustments, and
more transitions alone did not. PPO frequently failed to improve the bootstrap,
so candidate selection must compare both rather than assuming the final PPO
weights are better.

## Fixed-door native D1 lineage

This experiment implements option 3 from the architecture review: privileged
teacher, recurrent deployable actor, DAgger-style imitation, then
asymmetric-critic PPO. The actor receives `64x48` current/delta/motion channels,
21 deployable IMU/odometry/action/phase values, and recurrent state. It outputs
forward speed and yaw rate above stock firmware stabilization. The six-value
teacher/target tail is available only to the training teacher and critic.

| Version and change | Steps | Success | Collision | Outside-FOV success | Yaw p95 | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v45 stateful BC | BC | 0% | incomplete screen | 0% | 1.000 | Inert forward behavior |
| v46 20% DAgger | DAgger | 0% | 0% | 0% | 1.000 | Safe but inert |
| v47 21-value state BC | BC | 0% | 1.56% | 0% | 1.000 | State parity alone insufficient |
| v48 50% DAgger | DAgger | 0% | 7.81% | 0% | 1.000 | More imitation did not unlock completion |
| v49 exact phase BC | BC | 12.31% | 15.38% | 10.45% | 1.000 | First completion; phase is decisive |
| v50 conservative DAgger | DAgger | 0.78% | 0% | 0% | 0.996 | Safety/completion collapse |
| v51 stock recurrent PPO candidate | 1.05M | 15.38% | 51.75% | 20.83% | 0.958 | Rejected; v49 retained |
| v52 persistent asymmetric PPO | 262K | 33.59% | 20.31% | 34.85% | 0.570 | First material RL improvement |
| v53 conservative continuation | 262K | **34.11%** | **18.60%** | 28.79% | **0.465** | Current shadow candidate |

The stock Puffer continuation reached about 2,227 learner SPS, but its shared
critic and rollout-local recurrent state destabilized collision behavior. The
custom trainer keeps MinGRU state across 64-step rollouts, masks it only at
episode boundaries, gives privileged target state to a training-only critic,
and regularizes the on-policy actor toward teacher actions. This produced the
first large completion gain while reducing yaw saturation.

The result is useful but does not pass the pre-registered D1 simulation gate:
80% completion, at most 3% collision, and 70% outside-FOV completion. It also
does not yet prove fully student-visible phase inference. Simulation v49-v53
uses exact rendered phase labels, while the Mac runtime derives phase from
Grounding DINO detections and evidence age. These are host-assisted demo
checkpoints, not final end-to-end low-resolution semantic policies.

### Current host demo gate

`v53` is hash-bound to its training report and has a monitor-only live runner.
The runner concurrently reads AI Deck frames, Crazyflie telemetry, and
Grounding DINO detections; reconstructs the training observation; advances the
recurrent policy; and records proposed forward/yaw actions. It contains no arm,
takeoff, commander, or control path.

```bash
python3.13 scripts/crazyflie_door_puffer_shadow.py \
  --checkpoint artifacts/puffer_fixed_door_d1_v53_asymmetric_conservative262k/flightrl_fixed_door_d1_seed11_262144.bin \
  --prompt door \
  --threshold 0.20 \
  --duration-s 20
```

Dry-run checkpoint/report verification passes. A 30-frame QVGA door replay had
23 detector-positive frames, covered track/approach/recover phase transitions,
produced finite outputs, and measured compact-policy inference p95 below
`0.6 ms`. That replay used static telemetry, so the next evidence must be a real
stationary or firmware-held-hover shadow trace. Learned authority stays zero.

## Semantic MuJoCo lineage

MuJoCo performs native physics and rendering, but FlightRL currently loops
environments and frames through Python. The lineage combines `128x96` gray4
vision, a spatial map, MinGRU memory, a privileged teacher, semantic discovery,
exploration, approach behavior, and learned safety.

| Version and primary change | Steps | Train/eval rooms | Selected | Mission success | Discovery | Collision | Key result |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| v4 active exploration | 32K | 4/4 | bootstrap | 0% | 50% | 0% | Exploration alone insufficient |
| v5 full scan + clearance | 32K | 4/4 | PPO | 25% | 75% | 0% | First partial completion |
| v6 `128x96` | 32K | 4/4 | PPO | 25% | 75% | 0% | Resolution did not improve completion |
| v7 spatial memory | 32K | 4/4 | bootstrap | 60% | 80% | 0% | Best small-room result; weak generality evidence |
| v8 risk head | 32K | 4/4 | PPO | 0% | 75% | 0% | Auxiliary risk objective collapsed completion |
| v9 16 rooms | 32K | 16/4 | bootstrap | 0% | 25% | 0% | More diversity with fixed budget hurt |
| v10 longer diverse eval | 65K | 4/8 | PPO | 23.53% | 82.35% | 0% | Safe but incomplete |
| v11 12 diverse rooms | 131K | 12/8 | bootstrap | 38.89% | 88.89% | 5.56% | Best larger-run completion |
| v12 safer reset | 131K | 12/8 | bootstrap | 29.41% | 94.12% | 5.88% | Discovery improved, completion fell |
| v13 horizontal safety | 131K | 12/8 | bootstrap | 25% | 100% | 6.25% | False-safe and unsafe-forward regressed |
| v14 recurrent navigation safety | 131K | 12/8 | bootstrap | 0% | 56.25% | 12.5% | Coupled hidden state caused major regression |
| v15 stateful coverage | 131K | 12/8 | PPO | 0% | 75% | 6.25% | Coverage memory did not restore missions |
| v16 stateful Puffer path | 131K | 12/8 | bootstrap | 0% | 75% | 0% | Conservative but inert |
| v24 calibrated recurrent safety | 131K | 12/8 | PPO | 6.25% | 75% | 6.25% | Better proxy safety, poor task behavior |
| v26 memory-balanced imitation | 131K | 12/8 | bootstrap | 22.22% | 77.78% | 5.56% | Completion recovered, safety worsened |
| v27 frozen midpoint safety | 131K | 12/8 | PPO | 23.53% | 76.47% | 5.88% | No material safety gain |
| v28 balanced safety replay | 131K | 12/8 | PPO | 23.53% | 76.47% | 5.88% | Lower unsafe-forward, higher false-safe |

Versions 17-23 were implementation/probe iterations without comparable
decision-grade reports. A short v28 probe appeared substantially better than
the final eight-room evaluation. This is direct evidence that two-room probes
must not be used for checkpoint promotion.

### Semantic lesson

The problem is not proven to be solved by 50 instead of 12 rooms. More rooms
with the same transition budget already made v9 worse. The current experiment
entangles five hard problems:

- grounding the requested object;
- exploring when it is not visible;
- remembering visited space and target observations;
- avoiding obstacles from a forward monocular camera;
- approaching and holding at the object.

Safety labels also ask the forward camera to infer some side-clearance states
that may not be observable. Teacher imitation, PPO, and safety optimization can
therefore issue inconsistent gradients. The bootstrap often outperforms PPO,
which indicates that the current RL objective does not reliably preserve
teacher behavior.

## Native fixed-door teacher/student reset

The research-backed native semantic lane then isolated one mission: find an
interior door, center it, approach, and stop at a `0.80 m` standoff. The
teacher receives simulator geometry only to produce labels. The deployable
student receives the `64x48` current/delta/motion camera tensor, firmware
telemetry, mission phase, detector confidence/centroid/scale/staleness, its
previous executed action, and recurrent state.

| Version | Control initialization | Training | Success | Outside-FOV success | Collision | Masked-camera success | Decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Teacher upper bound | Deterministic detector teacher | None | 93.55% | 90.68% | 0.22% | N/A | Label generator, not deployable |
| v54 | v53 controller | PPO | 49.06% screen | 48.10% | 41.51% | N/A | Reject |
| v55 | v53 controller | BC | 2.34% | 4.55% | 1.56% | N/A | Reject |
| v57 | v53 controller | DAgger | 10.85% | 2.99% | 6.98% | N/A | Reject |
| v58 | v53 controller | BC | 7.03% | 1.52% | 7.03% | N/A | Reject |
| v59 | Fresh recurrent control, perception warm-start | BC, 1,048,576 samples | 79.11% | 74.73% | 0.56% | 1.53% | Real shadow candidate |

The v59 camera ablation is causal evidence: disabling both pixels and detector
evidence drops success from `79.11%` to `1.53%`. The important intervention was
not a larger room count or more PPO. Reusing the old controller trapped the
student in a poor behavior regime; keeping the perception encoder while
reinitializing fusion, recurrence, and action decoding produced the first
useful student.

For live transfer, v59 is connected only through firmware-stabilized yaw
authority. The initial envelope is translation zero, yaw at most `8 deg/s`,
and stale proposals stop. A v53 shadow trace cannot approve v59: checkpoint,
evaluation, shadow summary, and raw shadow CSV are hash-bound. The first v59
flight still requires a real no-control trace with finite actions, fresh door
detections, and correct yaw sign relative to the detection centroid.

One contract defect remains explicit: the fixed-door config declares
`max_yawrate_deg_s = 70`, but the v59 native step path does not consume that
field. It instead scales normalized yaw by the inherited physics value
`max_rate_yaw = 4 rad/s` (`229.18 deg/s`). The v59 host adapter therefore uses
the actual training scale when encoding the previously executed action, while
the authority envelope still clamps the real command to `8 deg/s`. The next
checkpoint should fix the native scale and retrain rather than silently
changing v59 semantics.

## Project evolution

1. **May: training and dynamics plumbing.** Planar and six-DoF policies,
   Puffer/Ocean integration, hover and waypoint tasks, and basic PPO/DAgger
   established the training path.
2. **June: range-based autonomy and sim-to-real gates.** Obstacle tasks, Flow
   and ranger telemetry, system identification, replay, and live gates were
   added. A direct raw-control policy accelerated into a crash. Analysis found
   a pitch-command sign mismatch and poor pre-contact drift coverage.
3. **June/July: control-boundary correction.** Firmware-stabilized setpoints and
   bounded residual authority became the live path. Direct motor/RPM policies
   remained simulation-only. The official Puffer drone environment was added
   as a reference lane, not silently treated as hardware-contract parity.
4. **July: AI Deck recovery and vision contract.** GAP8 bootloader recovery,
   frame-integrity fixes, gray4 transport, and synchronized camera/telemetry
   established a reliable `64x48` stream near 65 FPS.
5. **July 26-27: native visual breakthrough.** A native low-resolution renderer,
   recurrent expert bootstrap, and domain randomization produced strong causal
   local-control simulation results and a stationary live shadow pass.
6. **July 27-29: semantic expansion.** Object search, memory, teacher
   distillation, and safety were implemented in a separate MuJoCo/Python
   training stack. Completion and safety oscillated, throughput collapsed by
   several orders of magnitude, and no checkpoint passed the shadow gate.

## What worked

- Stock firmware stabilization with learned setpoint/residual authority.
- Native C environment stepping and rendering with Puffer vector collection.
- Small recurrent policies and recurrent expert bootstrap.
- Episode-level dynamics, room, exposure, texture, and sensor randomization.
- Masked/shuffled-camera causal checks instead of relying on PPO return.
- Separate simulation, replay, shadow, and bounded-live gates.
- AI Deck gray4 streaming and a configurable vision observation contract.
- Higher-resolution offboard grounding as development scaffolding.

## What failed or misled us

- Direct raw live control before command-sign and crash-distribution parity.
- Treating training return, short probes, or clear-room success as visual proof.
- Assuming more transitions or more rooms automatically improve generalization.
- Sequential one-seed tuning while changing room sets, labels, rewards, and
  selection metrics.
- Coupling navigation recurrence and safety recurrence.
- Building semantic training around a slow per-environment MuJoCo/Python loop
  instead of extending the proven native C scene path.
- Treating a learned world model as a substitute for explicit randomized rooms.
- Using one monolithic mission metric before validating grounding, exploration,
  safety, and approach independently.

## Architecture reset to evaluate

Do not implement this recommendation until the research task tests it against
published systems and available code:

1. Keep v5 low-light as the local visual-control baseline and v28 only as a
   semantic failure-analysis artifact.
2. Extend the native C renderer with fixed-capacity semantic primitives,
   distractors, target IDs, approach anchors, lighting, camera pose, latency,
   and frame-drop randomization.
3. Separate three contracts: semantic grounder/tracker, recurrent exploration
   and navigation actor, and deterministic firmware safety/authority envelope.
4. Train the navigation actor from target heatmap/bearing/confidence/staleness,
   not oracle world coordinates. Keep privileged state only in the teacher and
   critic when explicitly tested.
5. Gate each component independently before end-to-end missions.
6. Use MuJoCo or a stronger renderer as an independent fidelity/evaluation lane,
   not the default bulk-training loop.
7. Add learned latent dynamics or a world model only as a measured challenger:
   real-log residual modeling, representation learning, or imagined rollouts.

## Required benchmark before more tuning

Use a frozen matrix with at least three seeds and equal environment-transition
and optimizer-update budgets:

- procedural training room families plus unseen held-out families;
- held-out layouts, target instances, distractors, lighting, camera parameters,
  and command paraphrases;
- full camera, masked camera, shuffled temporal order, and oracle-target upper
  bound;
- grounder precision/recall and target reacquisition;
- exploration coverage and discovery time;
- collision, minimum clearance, unsafe-forward, and false-safe rates;
- approach/hold success and complete mission success;
- environment SPS, learner SPS, model parameters, MACs, and latency.

Promotion requires confidence intervals and no regression on the fixed native
local-control baseline. A larger scene count is justified only after a room
scaling curve shows which component is data-limited.

## Research brief for the continuation task

The continuation must use primary papers, official repositories, and actual
code rather than social-media summaries. It should compare:

- PufferLib `ocean/drone`, `tensaur/drone`, Emerge-Lab DroneRacing,
  PufferNet/dronelib, and Crazyflie firmware insertion examples;
- Bitcraze AI Deck/GAP8 examples, PULP-DroNet variants, Tiny-PULP-DroNet,
  NanoFlowNet, and ETH/PULP deployment tooling;
- visual drone control with privileged teachers, imitation, RL fine-tuning,
  recurrent policies, active vision, and low-resolution/foveated inputs;
- object-goal navigation and VLN systems that separate grounding, mapping,
  exploration, and control;
- explicit simulators and procedural scene generation across MuJoCo, Habitat,
  Isaac/Aerial Gym, Flightmare, AirSim, PyBullet drone environments, and other
  reproducible aerial stacks;
- learned world models for drones and robots, including whether they improve
  control, exploration, domain adaptation, or only representation/prediction.

For every candidate, record task, observation, action/control level, simulator,
hardware, model size, training throughput, sim-to-real evidence, code/license,
reproducibility, and direct reuse value. The output must recommend one primary
architecture, one challenger, a reuse plan, a fixed experiment matrix, and kill
criteria. PufferLib/native C remains a strong preference, not a predetermined
answer.

## Primary local evidence

- `docs/native_visual_training_fast_lane.md`
- `docs/research/vision_observation_contract_20260724.md`
- `docs/research/semantic_mission_architecture_20260726.md`
- `docs/research/pulp_dronet_2019_analysis.md`
- `.learnings/2026-06-22-direct-raw-live-failure.md`
- `.learnings/2026-07-08-vla-teacher-distillation.md`
- `.learnings/2026-07-18-vln-lite-ai-deck-readiness.md`
- `artifacts/puffer_visual/*.report.json`
- `artifacts/puffer_semantic/*.report.json`
