# Mac-to-AI-Deck roadmap

## End state

Train and verify a small target-conditioned indoor-navigation actor on the Mac,
then lower that same versioned actor to the AI Deck. The actor proposes bounded
velocity and yaw-rate setpoints. The Crazyflie STM32 remains authoritative for
state estimation, safety checks, stabilization, and motor mixing.

```text
Mac research                              Onboard runtime

native C / MuJoCo scenes                  AI Deck camera preprocessing
privileged teachers                       edge-v3 recurrent actor
RL / imitation / distillation      ->     bounded CPX proposal
held-out evaluation                              |
                                                  v
                                      STM32 safety + stabilizer
                                                  |
                                                  v
                                                motors
```

The current deployment contract is `aideck-navigation-policy-v3`. Pre-review
learned actor families are not intermediate deployment stages.

## Stage 1: Establish trustworthy desktop supervision

- Validate task, reset, reward, action, and success semantics with focused
  tests and independent native/MuJoCo checks where appropriate.
- Use privileged teachers to establish feasibility and generate labels.
- Keep task selection fixed within each episode and evaluation seeds disjoint.
- Bind reports to the clean source commit, exact metric/configuration, source
  hashes, dependency/runtime identities, and generated artifact hashes.

The fixed-door privileged teacher is useful evidence that its obstacle-free
approach/settle objective is solvable. It is not a learned policy, a general
navigation result, or a deployment candidate.

## Stage 2: Train the exact edge-v3 actor on the Mac

Implement one explicit adapter from simulator/teacher state and camera output
to the exact 64x48 gray4 plus 19-telemetry plus target-ID contract. Train or
distill the existing edge-shaped PyTorch reference and evaluate complete
recurrent sequences, including reset and invalid-input cases.

Promotion at this stage requires fresh held-out results for target-present,
target-absent, hard-negative, obstacle, lighting, latency/drop, and room-layout
variation. A desktop checkpoint remains non-deployable.

## Stage 3: Freeze and lower

Freeze and hash preprocessing, operator choices, tensor layouts, weights,
calibration data, rounding, saturation, and recurrent reset semantics. Then
establish:

1. PyTorch-float to host-float-C sequence parity;
2. calibrated int8 task-quality regression;
3. host-int8-C to GAP8 bit-exact sequence parity;
4. actual ELF L1/L2/stack/workspace fit;
5. sustained GAP8 latency under camera and communication load.

Static parameter, prospective byte, and MAC counts only support planning. They
do not prove embedded fit or speed.

## Stage 4: Build the command boundary

Define a byte-bound CPX proposal protocol with version, mission/reset epoch,
target identity, source frame sequence/time, proposal sequence, and finite
bounded outputs. The STM32 must reject stale, duplicate, reordered, malformed,
nonfinite, or contract-mismatched proposals without advancing applied-action
feedback.

Independent STM32 logic must enforce estimator/deck health, altitude/ranger
constraints, geofence, action clamps, slew limits, and a deadman. Actor boot,
mission/target change, estimator reset, invalid input, and excessive frame gap
must reset recurrent state.

## Stage 5: Promote hardware evidence gradually

The order is intentionally one-way:

1. grounded camera and telemetry capture;
2. synchronized offline replay;
3. on-device inference parity and timing with no proposals applied;
4. passive shadow against independent safety telemetry;
5. tethered bounded-axis proposal tests;
6. only then, separately authorized mission flights.

Every stage has its own abort criteria and produces evidence, not authority for
the next stage. Physical execution is never implied by a passing software gate.

## Current gaps

- exact edge-v3 observation/supervision adapter and trainer;
- fresh learned edge-v3 checkpoint and held-out evaluation;
- float-C, calibrated-int8, and GAP8 implementations;
- measured target memory/latency and recurrent sequence parity;
- CPX proposal transport and STM32 proposal-safety runtime;
- typed deployment bundle binding every binary, contract, and evidence input;
- positive learned-hardware approval path.

Until these are closed, the valid work surface is Mac simulation/training and
non-actuating hardware capture, telemetry, replay, and inference measurement.
