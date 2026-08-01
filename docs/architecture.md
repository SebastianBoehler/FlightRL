# Architecture

## System boundary

FlightRL has three deliberately separate layers:

```text
desktop research                    future deployed runtime

native C / MuJoCo environments      AI Deck GAP8
privileged teachers                 camera preprocessing
PPO / imitation / distillation  ->  edge-v3 recurrent actor
held-out challenge evaluation             |
                                             v CPX proposal
                                       STM32 safety layer
                                       estimator/stabilizer
                                             |
                                             v
                                           motors
```

The desktop layers may use privileged state, larger critics, and richer
diagnostics. The deployed actor may consume only the exact edge-v3 observation
contract. The STM32 must independently decide whether and how a proposal is
applied; the actor never owns motor authority.

## Edge-v3 actor

`aideck-navigation-policy-v3` is the only current deployment target. Its model
input is one current 64x48 gray4 frame, 19 normalized telemetry values, and a
closed-vocabulary target ID (`door`, `monitor`, or `sink`). Its outputs are
normalized body-frame `vx`/`vy`, world-up `vz`, and world-up yaw-rate proposals.

The reference implementation is split into:

- `puffer4_edge_contract.py`: units, frames, normalization, wire records,
  sequence/reset rules, target vocabulary, and action scales;
- `puffer4_edge_policy.py`: edge-shaped PyTorch visual/recurrent actor;
- `puffer4_edge_budget.py`: parameter, prospective quantized-byte, MAC, and
  activation estimates.

The PyTorch graph is a design reference. It becomes an exact deployment graph
only after preprocessing/operator freeze, float-C parity, calibrated int8
validation, GAP8 sequence parity, and measured target memory/latency.

## Desktop environments

### Native C

The local extension owns contiguous vector state and writes observations,
rewards, terminations, truncations, and episodic metrics in place. The generic
native simulator is divided by responsibility:

- `native_actions.c`: bounded action handling and actuator smoothing;
- `native_dynamics.c`: planar dynamics;
- `native_sixdof.c`: six-DoF dynamics/reward integration;
- `native_tasks.c`: task progression;
- `native_reward.c`: reward decomposition;
- `native_observation.c`: observation assembly;
- `native_reset.c`: reset and randomization;
- `native_termination.c`: crash, timeout, and bounds checks;
- `native_logging.c`: episodic aggregation;
- `binding.c` and binding headers: NumPy/Python vector interface.

The fixed-door environment has a separate generated Ocean binding and explicit
mission metric and privileged-teacher action contract. Its success metric
requires approach and settle at the configured standoff, pose and velocity
tolerances, continued visibility, and a 33-step hold. The retired fixed-door
student actor is not loaded or reinterpreted as edge-v3.

### MuJoCo

MuJoCo is an independent validation/calibration lane for rigid-body dynamics,
contacts, room geometry, sensor semantics, and orbit/circle behavior. It shares
task/reward contracts where exact parity is intended and rejects conflicting
explicit physics/control settings. Its current rate-lag approximation is not
claimed to be identical to native dynamics.

### PufferLib export

FlightRL can export native environments into a separate upstream PufferLib 4
checkout. The exporter copies the selected C modules, emits a thin Ocean
`binding.c`, and writes the corresponding `.ini`. PufferLib checkpoints remain
desktop research artifacts unless a later typed edge-v3 distillation and
deployment bundle binds them to the onboard graph.

## Episode and action semantics

Task identity is established before the first observation and remains stable
for the episode. Only completed environments receive a new task. Probability
vectors must be finite, nonnegative, correctly shaped, and have positive total
mass.

Continuous stochastic actors use a tanh-transformed normal distribution. PPO
stores/reuses the pre-tanh sample for log-probability evaluation; it does not
apply an inverse transform to an already saturated/clamped mode. Distribution
locations, scales, and bounds must be finite, and scales must be positive.

These desktop action contracts are not the edge-v3 setpoint ABI. Translation is
explicit and byte/contract bound; shape equality is never sufficient evidence
of semantic compatibility.

## Evidence and authority

Reports are evidence, not authority. Inputs that affect a claim must be bound by
resolved path and SHA-256 and parsed with exact JSON boolean semantics. A
record saying `ready=true` while also carrying failures is blocked.

The generic sim-to-real manifest intentionally emits zero hardware-approved
checkpoints. `require_hardware_approved()` unconditionally blocks learned live
control because no typed edge-v3 deployment bundle producer exists. This is the
correct current safety state, not an unfinished positive approval route.

A future deployment bundle must bind at least:

- source commit and dependency/toolchain identities;
- observation, action, mission, target-vocabulary, and model-format contracts;
- exact float/int8 weights, preprocessing, quantization, and tensor layouts;
- host/GAP8 recurrent-sequence parity and reset/error vectors;
- GAP8 ELF, firmware, CPX, STM32 safety configuration, and runtime identities;
- calibration/held-out evidence provenance and freshness.

## Hardware code

Hardware modules exist for Crazyflie connection/preflight, camera/firmware
capture, telemetry, nonlearned bring-up, calibration evidence, and
non-actuating grounding. They do not contain a learned edge-v3 flight runtime.

Physical work is staged independently from policy promotion. Camera or
telemetry evidence may be retained even when every learned checkpoint is
invalidated. Generated checkpoints/runs are not retained merely because they
were once successful.

## Current deliberate gaps

- no edge-v3 observation adapter, distillation/training entrypoint, or held-out
  student evaluator;
- no frozen float-C or calibrated int8 implementation;
- no GAP8 kernel, ELF memory proof, or sustained target latency measurement;
- no CPX proposal transport or STM32 proposal-safety implementation;
- no typed edge-v3 deployment bundle or positive learned-flight authority;
- no claim that the corrected fixed-door teacher covers arbitrary obstacles,
  lighting, latency, other room footprints, a learned student, or physical
  flight;
- no multi-drone stepping or swarm policy.

Unsupported paths fail closed instead of substituting legacy checkpoints,
partial state loading, mock data, or inferred authority.
