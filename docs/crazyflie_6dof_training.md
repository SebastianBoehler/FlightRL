# Crazyflie 6-DoF Training Track

This track is for simulation and replay work only. It is not cleared for live hardware control.

## Why This Exists

The original FlightRL native environment is intentionally fast and planar: `x/z/pitch`. That is useful for early PufferLib throughput, but it cannot represent yaw, side motion, full room geometry, or Multi-ranger beams in body coordinates.

The `flightrl.sixdof` package adds a port-ready 6-DoF reference surface:

- vectorized state arrays for many environments
- position, velocity, quaternion, and body-rate state
- six fixed body-frame ranger rays against a box room
- teacher policies for position/yaw, obstacle avoidance, attitude tracking, and circle flight
- Crazyflie-like CSV rollouts for replay and visualization

The code is written as a Python/Numpy spec first so behavior can be tested quickly before lowering the same structure into the native C/PufferLib path.

The first native lowering is available through `SixDofCrazyflieEnv(use_native_step=True)`, backed by `src/flightrl/native/native_sixdof.c`. It now handles the hot path in C: dynamics, raycasts, step counters, observation assembly, rewards, termination flags, and previous-action recording. It is exported with the PufferLib native files, but it is not yet a full standalone PufferLib/Ocean environment.

## Train Checkpoints

Small smoke run:

```bash
python scripts/train_sixdof_teacher.py \
  --task position_yaw \
  --updates 10 \
  --steps-per-update 16 \
  --num-envs 128
```

Longer overnight-style runs:

```bash
python scripts/train_sixdof_teacher.py --task position_yaw --updates 400 --num-envs 1024
python scripts/train_sixdof_teacher.py --task obstacle_avoidance --updates 400 --num-envs 1024
python scripts/train_sixdof_teacher.py --task attitude --updates 400 --num-envs 1024
python scripts/train_sixdof_teacher.py --task circle --updates 400 --num-envs 1024
```

Add `--native-step` to run the same training loop through the native C 6-DoF env hot path.

Default outputs:

- `artifacts/checkpoints/sixdof_position_yaw.pt`
- `artifacts/checkpoints/sixdof_obstacle_avoidance.pt`
- `artifacts/checkpoints/sixdof_attitude.pt`
- `artifacts/checkpoints/sixdof_circle.pt`

## Roll Out And Visualize

```bash
python scripts/rollout_sixdof_policy.py \
  --checkpoint artifacts/checkpoints/sixdof_obstacle_avoidance.pt \
  --steps 800 \
  --output artifacts/trajectories/sixdof_obstacle_avoidance.csv

python scripts/visualize_crazyflie_room.py \
  --input artifacts/trajectories/sixdof_obstacle_avoidance.csv \
  --output artifacts/trajectories/sixdof_obstacle_avoidance.room.png
```

The visualizer also works on real hardware logs that contain `stateEstimate.*`, `stabilizer.*`, and `range.*` columns.

## Compare Real And Sim

```bash
python scripts/compare_crazyflie_replay.py \
  --real artifacts/crazyflie_logs/ranger_hold_current_target_35s.csv \
  --sim artifacts/trajectories/sixdof_obstacle_avoidance.csv \
  --output artifacts/replay/hold_vs_sixdof_obstacle.json
```

This currently compares replay summaries, not exact trajectory matching. Exact replay alignment needs matching initial state, command interface, and timebase.

## Native Benchmark

```bash
python scripts/benchmark_sixdof_native.py --num-envs 8192 --steps 1000
```

This compares the Python vectorized spec, the raw native C dynamics/raycast kernel, and the native-backed environment hot loop. The benchmark is an implementation-health signal, not a policy-quality metric.

## PufferLib Export

```bash
python scripts/export_sixdof_puffer4.py --pufferlib-root /path/to/PufferLib
```

This writes:

- `ocean/flightrl_sixdof/binding.c`
- `ocean/flightrl_sixdof/native_sixdof.c`
- `ocean/flightrl_sixdof/native_sixdof.h`
- `config/flightrl_sixdof.ini`

The generated Ocean env is the first native PufferLib-oriented 6-DoF scaffold. It uses the same native hot loop as `SixDofCrazyflieEnv(use_native_step=True)`. The remaining work is building and training it inside a real upstream PufferLib checkout and adding exact replay/calibration gates.

## Hardware Boundary

Do not run these checkpoints directly on the Crazyflie. They are simulation-only.

The safe near-term hardware path remains:

1. firmware stabilizer stays enabled
2. deterministic ranger controller for live obstacle tests
3. learned policies stay in sim/replay until replay acceptance gates pass
4. later hardware deployment emits bounded setpoints with deadbands, acceleration limits, and watchdog aborts

Holding yaw while staying in place is realistic. Holding large fixed roll/pitch angles while staying at the same world position is physically constrained because tilted thrust creates horizontal acceleration.

## Morning Review Checklist

Before considering any live flight:

1. Inspect the rollout CSV and `.room.png` visualization for the checkpoint.
2. Confirm the rollout survived the intended horizon in simulation.
3. Compare against the latest real log with `scripts/compare_crazyflie_replay.py`.
4. Reject checkpoints that terminate early, saturate commands, or produce tiny ranger clearance.
5. Keep live tests on deterministic `MotionCommander` or reactive ranger control until a deployment gate exists.

The useful current artifact pattern is:

- `sixdof_position_yaw_h256_long.pt`: position/yaw simulation checkpoint candidate
- `sixdof_obstacle_avoidance_h256_long.pt`: obstacle-aware simulation checkpoint candidate
- `sixdof_attitude_h256_long.pt`: experimental attitude checkpoint; expected to fail position-hold goals
- `sixdof_circle_h256_long.pt`: experimental circle checkpoint; requires more closed-loop training
