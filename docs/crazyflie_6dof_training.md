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
python scripts/train_sixdof_teacher.py --task multitask --updates 400 --num-envs 1024
```

Add `--native-step` to run the same training loop through the native C 6-DoF env hot path.

Default outputs:

- `artifacts/checkpoints/sixdof_position_yaw.pt`
- `artifacts/checkpoints/sixdof_obstacle_avoidance.pt`
- `artifacts/checkpoints/sixdof_attitude.pt`
- `artifacts/checkpoints/sixdof_circle.pt`
- `artifacts/checkpoints/sixdof_multitask.pt`

The `multitask` checkpoint is task-conditioned: the model sees the base 28-value 6-DoF observation plus a one-hot task vector for position/yaw, obstacle avoidance, attitude, or circle behavior.

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

## Checkpoint Gates

```bash
python scripts/build_sixdof_teacher_dataset.py \
  --task position_yaw,obstacle_avoidance,circle \
  --native-step \
  --output artifacts/datasets/sixdof_teacher_safe_tasks.npz

python scripts/train_sixdof_offline.py \
  --dataset artifacts/datasets/sixdof_teacher_safe_tasks.npz \
  --checkpoint artifacts/checkpoints/sixdof_safe_tasks_offline.pt \
  --native-step

python scripts/evaluate_sixdof_action_gap.py \
  --checkpoint artifacts/checkpoints/sixdof_safe_tasks_offline.pt \
  --dataset artifacts/datasets/sixdof_teacher_safe_tasks.npz \
  --output artifacts/replay/sixdof_safe_tasks_offline_action_gap.json

python scripts/build_sixdof_dagger_dataset.py \
  --checkpoint artifacts/checkpoints/sixdof_safe_tasks_offline.pt \
  --append-dataset artifacts/datasets/sixdof_teacher_safe_tasks.npz \
  --native-step \
  --output artifacts/datasets/sixdof_safe_tasks_dagger.npz

python scripts/train_sixdof_offline.py \
  --dataset artifacts/datasets/sixdof_safe_tasks_dagger.npz \
  --checkpoint artifacts/checkpoints/sixdof_safe_tasks_dagger.pt \
  --native-step

python scripts/train_sixdof_dagger.py \
  --seed-dataset artifacts/datasets/sixdof_teacher_safe_tasks.npz \
  --initial-checkpoint artifacts/checkpoints/sixdof_safe_tasks_offline.pt \
  --output-dir artifacts/dagger/sixdof_safe_tasks \
  --iterations 3 \
  --native-step

python scripts/evaluate_sixdof_checkpoint.py \
  --teacher \
  --task position_yaw,obstacle_avoidance,circle \
  --native-step \
  --output artifacts/replay/sixdof_teacher_safe_tasks_gate.json

python scripts/evaluate_sixdof_checkpoint.py \
  --checkpoint artifacts/checkpoints/sixdof_multitask_h256.pt \
  --native-step \
  --output artifacts/replay/sixdof_multitask_h256_gate.json

python scripts/evaluate_sixdof_checkpoint.py \
  --checkpoint artifacts/dagger/sixdof_obstacle_focus_refine/iter_02.pt \
  --task obstacle_avoidance \
  --native-step \
  --output artifacts/replay/sixdof_obstacle_focus_refine_gate.json
```

The gate checks low-percentile simulated wall clearance, terminal-free completion fraction, and mean position error. It also reports control diagnostics: action magnitude, saturation fraction, and learned-policy disagreement with the analytic teacher on the states visited by the policy. A pass is only a simulation acceptance signal; it does not approve live Crazyflie deployment.

Use the teacher gate first. If the analytic teacher fails a task, a learned checkpoint for that same task is not meaningful yet. Current evidence shows position/yaw, obstacle avoidance, and circle are feasible as a reference set, while the experimental attitude task needs a better physical objective before it belongs in multi-task training.

Use the offline action-gap report before closed-loop gates. If a policy cannot match teacher actions on teacher-visited states, closed-loop rollout failures are expected and more DAgger/RL training is premature.

If teacher-state imitation has a low action gap but fails closed-loop, collect DAgger data with `build_sixdof_dagger_dataset.py`. That script rolls out the checkpoint, records policy-visited states, labels those states with the analytic teacher, and optionally prepends an existing compatible dataset. This directly targets distribution shift between the teacher rollouts and states induced by the learned policy. Use `train_sixdof_dagger.py` for repeated collect/train/evaluate iterations; it writes per-iteration datasets, checkpoints, gate reports, and a `summary.json`.

For task-conditioned checkpoints, pass `--task` to evaluate a specific task without changing the checkpoint's one-hot task encoding. This is useful when one task has been refined through DAgger and the full multi-task gate is expected to remain stricter.

Training uses the same 800-step horizon for checkpoint selection by default. For CI smoke tests or quick experiments, reduce it explicitly with `--eval-steps`.

## Native Benchmark

```bash
python scripts/benchmark_sixdof_native.py --num-envs 8192 --steps 1000
python scripts/benchmark_sixdof_sweep.py \
  --env-counts 1024 4096 8192 16384 \
  --steps 500 \
  --output artifacts/replay/sixdof_native_benchmark_sweep.json
```

This compares the Python vectorized spec, the raw native C dynamics/raycast kernel, and the native-backed environment hot loop. The benchmark is an implementation-health signal, not a policy-quality metric.

On the current Apple Silicon development machine, the sweep peaked at `16,030,292` native env steps/sec with `4096` envs. The raw native kernel stayed around `15.0M-16.5M` steps/sec in the 1024-16384 env range, while the Python vectorized spec was around `0.7M-1.1M` steps/sec. The practical starting point for native/Puffer sizing is therefore `4096` total agents.

To summarize a checkpoint candidate after gate, action-gap, and edge export:

```bash
python scripts/summarize_sixdof_artifact.py \
  --name sixdof_obstacle_focus_refine \
  --checkpoint artifacts/dagger/sixdof_obstacle_focus_refine/iter_02.pt \
  --gate artifacts/replay/sixdof_obstacle_focus_refine_gate.json \
  --action-gap artifacts/replay/sixdof_obstacle_focus_refine_action_gap.json \
  --edge-parity artifacts/edge/sixdof_obstacle_focus_refine.parity.json \
  --output artifacts/replay/sixdof_obstacle_focus_refine_summary.json
```

## PufferLib Export

```bash
python scripts/export_sixdof_puffer4.py --pufferlib-root /path/to/PufferLib
```

This writes:

- `ocean/flightrl_sixdof/binding.c`
- `ocean/flightrl_sixdof/native_sixdof.c`
- `ocean/flightrl_sixdof/native_sixdof.h`
- `config/flightrl_sixdof.ini`

The generated Ocean env is the first native PufferLib-oriented 6-DoF scaffold. It uses the same native hot loop as `SixDofCrazyflieEnv(use_native_step=True)`.

To build and run a small CPU/PyTorch-backend Puffer smoke train:

```bash
python scripts/train_sixdof_puffer4.py \
  --pufferlib-root /path/to/PufferLib \
  --env-name flightrl_sixdof \
  --total-agents 1024 \
  --num-buffers 1 \
  --build-mode cpu \
  -- --train.total-timesteps 32768 --train.horizon 16 --train.minibatch-size 1024
```

On macOS, the CPU backend is run with `OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=TRUE` by default. Without that guard, importing the Puffer Python training stack can collide with already-loaded OpenMP runtimes from Torch/Numpy and crash during `cpu_step`.

Current CPU Puffer smoke results are much lower than the raw env benchmark because Torch forward/backward dominates the short PPO runs. The best measured mixed train throughput from the quick sweep was about `430K-445K` SPS with `4096` agents, `8` buffers, `8` threads, horizon `32`, minibatch `16384`, and replay ratio `1`. Replay ratio `2` with the same agent count measured about `245K-253K` SPS. Eval-only dashboard samples after training reported about `1.2M-1.4M` SPS, so the next optimization target is policy/training overhead, not native dynamics.

To generate or execute the reproducible Puffer sweep matrix:

```bash
python scripts/run_sixdof_puffer_sweep.py \
  --output artifacts/replay/sixdof_puffer_sweep_manifest.json

python scripts/run_sixdof_puffer_sweep.py \
  --run \
  --no-build \
  --env-name flightrl_sixdof_sweep_512 \
  --pufferlib-root ../PufferLib-4-flightrl \
  --total-timesteps 524288 \
  --output artifacts/replay/sixdof_puffer_sweep_smoke.json
```

The sweep varies total agents, buffer/thread count, horizon, minibatch size, replay ratio, learning rate, entropy, and policy hidden size. It reports train-only SPS separately from eval-only dashboard samples.

Recommended next Puffer tuning baseline:

```bash
python scripts/train_sixdof_puffer4.py \
  --pufferlib-root ../PufferLib-4-flightrl \
  --env-name flightrl_sixdof_sweep_512 \
  --total-agents 4096 \
  --num-buffers 8 \
  --num-threads 8 \
  --build-mode cpu \
  --no-build \
  -- \
  --train.total-timesteps 2097152 \
  --train.horizon 32 \
  --train.minibatch-size 16384 \
  --train.replay-ratio 1 \
  --train.learning-rate 0.0007 \
  --train.ent-coef 0.003
```

Treat Puffer `.bin` checkpoints as native trainer artifacts until an explicit import/evaluation bridge is added. The hardware-facing checkpoint candidate remains the gated Torch checkpoint summary above, not a short Puffer smoke checkpoint.

## Edge Export Contract

Issue #12 tracks the tiny-model deployment path. The current first contract is a TorchScript trace for simulation checkpoints:

```bash
python scripts/export_sixdof_edge_policy.py \
  --checkpoint artifacts/checkpoints/sixdof_position_yaw_h256_long.pt \
  --output artifacts/edge/sixdof_position_yaw_h256_long.ts
```

The export writes a `.parity.json` report with:

- observation shape and dtype: `[28]`, `float32`
- action shape and dtype: `[4]`, `float32`
- action bounds: `[-1, 1]`
- action meaning: thrust, roll-rate, pitch-rate, yaw-rate
- max/mean absolute parity error between the Python policy and exported artifact

This is still a local inference smoke path, not an onboard deployment path. Hardware use remains gated by replay comparison, latency checks, and a flight safety envelope.

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
