# Crazyflie six-DoF desktop research

This lane develops and validates dynamics, tasks, rewards, teachers, and
learning procedures on the Mac. Its observations and actions are simulator
contracts; they are not the AI Deck ABI and its checkpoints cannot control a
Crazyflie.

The deployment target is `aideck-navigation-policy-v3`, documented in
`docs/edge_navigation_v3.md`. Moving knowledge from this lane to that actor
requires an explicit adapter/distillation dataset and an exact edge-v3
evaluation. Matching tensor shapes or successful simulator episodes are not a
conversion procedure.

## Environments

`flightrl.sixdof` provides the vectorized Python reference and a native C hot
path. Both represent position, velocity, attitude, body rate, ranger rays,
actuator lag, task state, reward, and episode termination. Use the Python lane
for readable reference behavior and the native lane for scale.

MuJoCo is an independent desktop validation lane for rigid-body dynamics,
contacts, room geometry, sensor semantics, and selected checkpoint replay. It
does not make native or learned behavior physically validated by itself.

Task identity is fixed for an episode. Keep training, selection, and final
evaluation seeds disjoint, and record the reset, physics, sensor, reward, and
task contracts with every result.

## Fast health checks

Build the native extension before native comparisons:

```bash
python setup.py build_ext --inplace --force
python scripts/benchmark_sixdof_native.py --num-envs 8192 --steps 1000
python scripts/benchmark_mujoco_sixdof.py --env-counts 1 8 32 --steps 300
```

Throughput is an implementation-health measurement, not policy evidence.
Never carry a historical throughput number forward after code, compiler,
machine, or environment settings change.

## Teacher and simulation policy workflow

First prove that the analytic teacher can satisfy the exact simulation gate:

```bash
python scripts/evaluate_sixdof_checkpoint.py \
  --teacher \
  --task position_yaw,obstacle_avoidance,circle \
  --native-step \
  --seed 10011 \
  --fail-on-gate \
  --output artifacts/evidence/sixdof_teacher_seed10011.json
```

Then create a fresh imitation candidate rather than loading an invalidated
pre-review checkpoint:

```bash
python scripts/train_sixdof_teacher.py \
  --task position_yaw,obstacle_avoidance,circle \
  --native-step \
  --seed 11011 \
  --checkpoint artifacts/experiments/sixdof_imitation_seed11011.pt

python scripts/evaluate_sixdof_checkpoint.py \
  --checkpoint artifacts/experiments/sixdof_imitation_seed11011.pt \
  --native-step \
  --seed 12011 \
  --fail-on-gate \
  --output artifacts/evidence/sixdof_imitation_seed12011.json
```

Closed-loop PPO and DAgger remain desktop research tools:

```bash
python scripts/train_sixdof_ppo.py \
  --task position_yaw \
  --native-step \
  --seed 13011 \
  --selection-seed 13012 \
  --evaluation-seed 13013 \
  --checkpoint artifacts/experiments/sixdof_ppo_seed13011.pt

python scripts/build_sixdof_dagger_dataset.py --help
python scripts/train_sixdof_dagger.py --help
```

Do not infer progress from training return alone. At minimum compare completion,
collisions/termination, clearance, position/yaw error, action magnitude,
saturation, and behavior under held-out reset/sensor/physics profiles.

## PufferLib lane

PufferLib is a separate checkout. Export a fresh environment from the current
source before using it:

```bash
python scripts/export_sixdof_puffer4.py \
  --pufferlib-root ../PufferLib-4-flightrl \
  --env-name flightrl_sixdof_reviewed
```

`scripts/train_sixdof_puffer4.py` and the sweep scripts are desktop training
tools. Their `.bin` outputs are Puffer trainer artifacts, not edge-v3 weights.
They are intentionally rejected by current checkpoint loaders: a raw
same-shaped state dict does not prove which task, observation, action, or
simulator contract produced it. Do not relabel or import one. Before Puffer
training can produce evaluable candidates, its producer must emit the current
checkpoint envelope itself and bind that envelope to the exact exported source
and configuration. It would still require a separate exact edge-v3 distillation
boundary before contributing to the onboard actor.

## Desktop export

TorchScript export exists only for local CPU parity and latency checks:

```bash
python scripts/export_sixdof_desktop_policy.py \
  --checkpoint artifacts/experiments/sixdof_imitation_seed11011.pt \
  --output artifacts/desktop/sixdof_imitation_seed11011.ts
```

The parity report does not prove int8 quality, GAP8 operator support, target
memory fit, target latency, CPX integrity, or STM32 safety behavior.

## Promotion boundary

A six-DoF result is useful when it improves the teacher, dynamics, curriculum,
or supervision data for the exact edge actor. It is not useful to accumulate
more incompatible checkpoint families.

Before any learned hardware proposal is considered, the repository still needs:

1. an exact six-DoF/fixed-door-to-edge-v3 observation and supervision adapter;
2. a fresh edge-v3 student trainer and held-out evaluator;
3. float-C, calibrated-int8, and GAP8 recurrent-sequence parity;
4. measured GAP8 ELF memory and sustained latency;
5. a sequence/freshness-bound CPX proposal protocol;
6. independent STM32 clamps, slew limits, estimator/geofence checks, and
   deadman behavior;
7. capture, replay, passive-shadow, then tethered bounded-axis gates.

Until all applicable gates exist and pass, remain on desktop simulation or
non-actuating capture/replay.
