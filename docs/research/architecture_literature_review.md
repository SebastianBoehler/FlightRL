# Architecture literature review reconciled with edge-v3

The source review was performed on 2026-07-29. Its local checkpoint rankings,
run metrics, staged live suggestions, and D-series actor plan were invalidated
by the first repository review and are intentionally not reproduced here. Git
history preserves the original snapshot for historical analysis.

This file retains only the literature conclusions that remain applicable to
`aideck-navigation-policy-v3`. It is not an experiment result, implementation
specification, or hardware authorization record.

## Current architecture decision

Use one small target-conditioned recurrent actor on the AI Deck to propose
bounded local navigation setpoints. Keep mission supervision and all flight
safety outside that actor:

```text
mission target: door / monitor / sink
                   |
AI Deck gray4 + telemetry -> recurrent edge-v3 actor
                                      |
                             bounded CPX proposal
                                      v
STM32 freshness / clamps / slew / estimator / geofence / deadman
                                      |
                           stock stabilizer and mixer
```

Mac-only environments may use privileged geometry, analytic teachers,
asymmetric critics, richer rendering, and large-scale rollouts. The deployed
student may consume only its exact versioned observation. Shape-compatible
desktop checkpoints are not edge policy parents.

The retained fixed-door lane is a privileged teacher/metric, not a learned
student. Its approach/settle success can establish supervision feasibility but
cannot establish perception, general navigation, edge execution, or hardware
readiness.

## Evidence that supports this split

| Reference | Durable lesson | Boundary for FlightRL |
| --- | --- | --- |
| [PULP-DroNet](https://arxiv.org/abs/1905.04166) | Compact GAP8 vision can produce bounded navigation intent above firmware stabilization. | Different task, input, outputs, and graph; no direct checkpoint reuse. |
| [Tiny-PULP-Dronet v3](https://arxiv.org/abs/2407.12675) | Very small quantized grayscale networks can be practical on GAP8-class hardware. | Published memory/rate does not prove edge-v3 ELF fit, operators, latency, or quality. |
| [Local/global nano-UAV perception](https://arxiv.org/abs/2403.11661) | Global visual intent and local geometric safety can be separate. | Extra ToF hardware is not assumed; independent range veto belongs to STM32 safety if implemented. |
| [RL bootstrapped by imitation](https://arxiv.org/abs/2403.12203) | A privileged RL teacher, visual imitation student, and adaptive RL fine-tuning can outperform either route alone. | Racing observations/actions and network scale do not transfer; test whether each stage earns its complexity. |
| [Deep Drone Acrobatics](https://github.com/uzh-rpg/deep_drone_acrobatics) | Temporal visual abstraction and iterative imitation can transfer through a lower-level controller. | Aggressive maneuvers and low-level action authority are out of scope. |
| [SINGER](https://openreview.net/forum?id=GyzYVEq4nO) | A language/semantic goal can condition closed-loop visual navigation learned from an expert. | Its compute, simulator, and evaluation do not establish nano-UAV feasibility or edge-v3 quality. |
| [PufferLib drone](https://github.com/PufferAI/PufferLib/tree/master/ocean/drone) | Native vector environments and compact trainer/runtime interfaces are useful. | Privileged observation and direct motor/RPM actions are incompatible with edge-v3. |
| [Crazyflow](https://github.com/utiasDSL/crazyflow) | Fitted batched Crazyflie dynamics/controllers can inform independent parameter tests. | Physics throughput is not rendered learner throughput; units and contracts require reconciliation. |
| [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) | Firmware/controller integration scenarios can provide independent regression cases. | It is a validation reference, not the Mac bulk-training or onboard runtime. |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | Rigid-body/contact and geometry checks are useful independent of the native fast path. | Agreement between two simulators still is not physical validation. |

## Learning approach

The literature supports a measured sequence, not a mandatory stack:

1. Prove the privileged teacher under the exact task metric.
2. Generate student-visible image/telemetry observations and teacher labels.
3. Train the exact edge-v3 recurrent student with disjoint selection and final
   evaluation seeds.
4. Add student-visited DAgger data only when distribution shift is measured.
5. Add RL/asymmetric-critic refinement only when imitation has reached its
   defined ceiling and the refined policy improves held-out closed-loop gates.

Auxiliary visibility/box/collision heads may shape representation during Mac
training, but exported inputs and outputs remain exactly edge-v3. Simulator
target pose, bearing, visibility truth, mission phase, and teacher action never
become hidden deployment inputs.

## Evaluation requirements

At minimum evaluate target-present, target-absent, physically verified hard
negative, initially outside-FOV, obstacle, lighting, blur/motion, room-layout,
sensor perturbation, and stale/drop/reorder conditions. Counterfactual target
token swaps in the same scene are required to show that conditioning changes
behavior instead of merely adding a class bias.

Report environment stepping separately from learner throughput and inference
latency. A simulator success rate does not replace masked-input leakage checks,
per-factor results, collision/clearance, action saturation/smoothness, or
complete recurrent-sequence error tests.

## Deployment requirements

The PyTorch actor's parameter and MAC counts make the direction plausible, not
deployable. Promotion still requires:

- frozen and hashed preprocessing, operators, layouts, weights, calibration,
  rounding, and reset semantics;
- PyTorch-float to host-float-C recurrent-sequence parity;
- calibrated int8 task-quality regression;
- host-int8-C to GAP8 bit-exact recurrent-sequence parity;
- measured GAP8 ELF L1/L2/stack/workspace and sustained latency;
- sequence/freshness-bound CPX records and independent STM32 rejection/safety;
- capture, replay, and passive-shadow evidence before any separately approved
  bounded physical test.

No current learned checkpoint satisfies this list. The correct near-term work
is the exact Mac student and lowering path, not another parallel actor family,
world model, dense mapper, VLM, VIO rewrite, or direct motor policy.
