# 6-DoF Puffer Sweep Baseline

Date: 2026-05-20

Machine: local Apple Silicon development machine.

## Native Env Baseline

Command:

```bash
python scripts/benchmark_sixdof_sweep.py \
  --env-counts 1024 4096 8192 16384 \
  --steps 500 \
  --output artifacts/replay/sixdof_native_benchmark_sweep.json
```

Result:

| envs | python SPS | native raw SPS | native env SPS |
| ---: | ---: | ---: | ---: |
| 1024 | 704493 | 15028784 | 14977420 |
| 4096 | 1016324 | 16456689 | 16030292 |
| 8192 | 1104672 | 15888712 | 15493667 |
| 16384 | 1121803 | 14952110 | 14487611 |

Best native env throughput was `16,030,292` steps/sec at `4096` envs.

## Puffer CPU Smoke Sweep

Command:

```bash
python scripts/run_sixdof_puffer_sweep.py \
  --run \
  --no-build \
  --max-variants 2 \
  --env-name flightrl_sixdof_sweep_512 \
  --pufferlib-root ../PufferLib-4-flightrl \
  --total-timesteps 524288 \
  --output artifacts/replay/sixdof_puffer_sweep_smoke.json
```

Result:

| variant | agents | buffers | threads | horizon | minibatch | replay | hidden | train SPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small_h16_rr2_h64 | 1024 | 1 | 1 | 16 | 2048 | 2 | 64 | 346600 |
| base_h32_rr2_h128 | 4096 | 8 | 8 | 32 | 8192 | 2 | 128 | 243000 |

Manual runs before the sweep runner showed about `430K-445K` train SPS for `4096` agents with replay ratio `1`, minibatch `16384`, learning rate `0.0007`, entropy `0.003`, and hidden size `128`. That remains the recommended next tuning baseline.

Full bounded CPU smoke sweep:

```bash
python scripts/run_sixdof_puffer_sweep.py \
  --run \
  --no-build \
  --env-name flightrl_sixdof_sweep_512 \
  --pufferlib-root ../PufferLib-4-flightrl \
  --total-timesteps 524288 \
  --output artifacts/replay/sixdof_puffer_sweep_full_smoke.json
```

| variant | agents | buffers | threads | horizon | minibatch | replay | lr | entropy | hidden | train SPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small_h16_rr2_h64 | 1024 | 1 | 1 | 16 | 2048 | 2 | 0.0003 | 0.001 | 64 | 397000 |
| base_h32_rr2_h128 | 4096 | 8 | 8 | 32 | 8192 | 2 | 0.0003 | 0.001 | 128 | 276600 |
| fast_h32_rr1_h128 | 4096 | 8 | 8 | 32 | 16384 | 1 | 0.0007 | 0.003 | 128 | 471000 |
| wide_h32_rr1_h256 | 4096 | 8 | 8 | 32 | 16384 | 1 | 0.0005 | 0.002 | 256 | 268300 |
| large_h32_rr1_h128 | 8192 | 8 | 8 | 32 | 16384 | 1 | 0.0003 | 0.001 | 128 | 474900 |

Result: replay ratio `1` with hidden size `128` is still the throughput baseline. The `8192`-agent run reached `474.9K` train SPS, narrowly ahead of the `4096`-agent fast run at `471.0K`. Hidden size `256` roughly halved train throughput for this short run.

## Current Refresh

Native simulator refresh:

```bash
python scripts/benchmark_sixdof_sweep.py \
  --env-counts 2048 4096 8192 16384 \
  --steps 500 \
  --output artifacts/replay/sixdof_native_benchmark_latest_recovery_run.json
```

| envs | python SPS | native raw SPS | native env SPS |
| ---: | ---: | ---: | ---: |
| 2048 | 870618 | 17225199 | 16153914 |
| 4096 | 1039572 | 16947002 | 16604104 |
| 8192 | 1101787 | 15533642 | 15047310 |
| 16384 | 1130103 | 14699517 | 14398359 |

Best native env throughput in this run was `16,604,104` steps/sec at `4096` envs.

Puffer refresh used the compiled env name `flightrl_sixdof_sweep`. A first attempt with `flightrl_sixdof_sweep_512` failed because the Puffer checkout had been built for `flightrl_sixdof_sweep`.

```bash
python scripts/run_sixdof_puffer_sweep.py \
  --run \
  --no-build \
  --env-name flightrl_sixdof_sweep \
  --pufferlib-root ../PufferLib-4-flightrl \
  --total-timesteps 524288 \
  --output artifacts/replay/sixdof_puffer_sweep_recovery_run_full.json
```

| variant | agents | buffers | threads | horizon | minibatch | replay | lr | entropy | hidden | train SPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small_h16_rr2_h64 | 1024 | 1 | 1 | 16 | 2048 | 2 | 0.0003 | 0.001 | 64 | 309000 |
| base_h32_rr2_h128 | 4096 | 8 | 8 | 32 | 8192 | 2 | 0.0003 | 0.001 | 128 | 220900 |
| fast_h32_rr1_h128 | 4096 | 8 | 8 | 32 | 16384 | 1 | 0.0007 | 0.003 | 128 | 372400 |
| wide_h32_rr1_h256 | 4096 | 8 | 8 | 32 | 16384 | 1 | 0.0005 | 0.002 | 256 | 230000 |
| large_h32_rr1_h128 | 8192 | 8 | 8 | 32 | 16384 | 1 | 0.0003 | 0.001 | 128 | 377700 |

Result: the current fastest Puffer setting is still replay ratio `1`, hidden size `128`. In this refresh, `8192` agents reached `377.7K` train SPS and the `4096`-agent fast run reached `372.4K`.

Follow-up: the Puffer runtime now preflights `--no-build` runs by querying the compiled native extension's `env_name`. If the checkout is built for a different env, it fails before launching training with a direct message to rebuild or use the compiled `--env-name`.

Guard check:

```bash
python scripts/run_sixdof_puffer_sweep.py \
  --run \
  --no-build \
  --max-variants 1 \
  --env-name flightrl_sixdof_sweep_512 \
  --pufferlib-root ../PufferLib-4-flightrl \
  --total-timesteps 65536 \
  --output artifacts/replay/sixdof_puffer_sweep_mismatch_guard.json
```

Result: the sweep record failed fast with `PufferLib native extension is built for 'flightrl_sixdof_sweep', not 'flightrl_sixdof_sweep_512'; rerun without --no-build or use the compiled --env-name.`

## Thread Scaling Refresh

The sweep runner now supports named variant subsets with `--variants`, records a `summary.best_train_sps`, and includes explicit 4/8/12-thread variants plus a horizon-64 variant for bounded tuning passes.

Command:

```bash
python scripts/run_sixdof_puffer_sweep.py \
  --run \
  --no-build \
  --variants fast_h32_t4_rr1_h128 fast_h32_rr1_h128 fast_h32_t12_rr1_h128 \
  --env-name flightrl_sixdof_sweep \
  --pufferlib-root ../PufferLib-4-flightrl \
  --total-timesteps 262144 \
  --output artifacts/replay/sixdof_puffer_thread_sweep_latest.json
```

| variant | agents | threads | horizon | replay | hidden | train SPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fast_h32_t4_rr1_h128 | 4096 | 4 | 32 | 1 | 128 | 394800 |
| fast_h32_rr1_h128 | 4096 | 8 | 32 | 1 | 128 | 451200 |
| fast_h32_t12_rr1_h128 | 4096 | 12 | 32 | 1 | 128 | 459600 |

Current short-run result: `fast_h32_t12_rr1_h128` is the fastest of this thread-scaling subset at `459.6K` train SPS, only slightly ahead of 8 threads. The next longer run should compare 8 vs 12 threads at a larger timestep budget before treating 12 threads as the default.

## Thread Scaling Confirmation

Command:

```bash
python scripts/run_sixdof_puffer_sweep.py \
  --run \
  --variants fast_h32_rr1_h128 fast_h32_t12_rr1_h128 \
  --total-timesteps 1048576 \
  --env-name flightrl_sixdof_sweep \
  --pufferlib-root ../PufferLib-4-flightrl \
  --output artifacts/replay/sixdof_puffer_thread_refresh_2026-05-20.json
```

| variant | agents | threads | horizon | replay | hidden | train SPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fast_h32_rr1_h128 | 4096 | 8 | 32 | 1 | 128 | 462800 |
| fast_h32_t12_rr1_h128 | 4096 | 12 | 32 | 1 | 128 | 474100 |

Result: 12 threads remains the current CPU throughput baseline for this machine at the same timestep budget, but the margin over 8 threads is modest. Keep `replay_ratio=1`, `hidden=128`, `horizon=32`, `minibatch=16384`, `lr=7e-4`, and `entropy=0.003` as the next Puffer tuning baseline.

Native simulator refresh:

```bash
python scripts/benchmark_sixdof_sweep.py \
  --env-counts 1024 4096 8192 16384 \
  --steps 400 \
  --output artifacts/replay/sixdof_native_benchmark_recovery_refresh.json
```

Best native env throughput was `17,390,135` steps/sec at `1024` envs; `4096` envs reached `16,368,838` env steps/sec.

## Full Short Sweep Refresh

Native PPO path:

```bash
python scripts/benchmark_sixdof_training_throughput.py \
  --variants smoke_64x16_h64 base_256x32_h128 large_512x32_h128 \
  --output artifacts/replay/sixdof_training_throughput_refresh_2026-05-20.json
```

| variant | envs | horizon | hidden | collect SPS | update SPS | total SPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke_64x16_h64 | 64 | 16 | 64 | 327855 | 305466 | 158132 |
| base_256x32_h128 | 256 | 32 | 128 | 725662 | 482681 | 289871 |
| large_512x32_h128 | 512 | 32 | 128 | 586078 | 595141 | 295287 |

Best local PyTorch PPO throughput is currently `large_512x32_h128` at `295,287` total samples/sec.

Puffer short sweep:

```bash
python scripts/run_sixdof_puffer_sweep.py \
  --run \
  --no-build \
  --total-timesteps 262144 \
  --output artifacts/replay/sixdof_puffer_full_short_2026-05-20.json
```

| variant | agents | threads | horizon | replay | hidden | train SPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small_h16_rr2_h64 | 1024 | 1 | 16 | 2 | 64 | 353600 |
| fast_h32_rr1_h128 | 4096 | 8 | 32 | 1 | 128 | 463600 |
| fast_h32_t4_rr1_h128 | 4096 | 4 | 32 | 1 | 128 | 472800 |
| fast_h32_t12_rr1_h128 | 4096 | 12 | 32 | 1 | 128 | 453600 |
| wide_h32_rr1_h256 | 4096 | 8 | 32 | 1 | 256 | 267700 |
| large_h32_rr1_h128 | 8192 | 8 | 32 | 1 | 128 | 472500 |
| large_h32_t12_rr1_h128 | 8192 | 12 | 32 | 1 | 128 | 466900 |
| long_h64_rr1_h128 | 4096 | 8 | 64 | 1 | 128 | 510800 |

Result: for short Puffer training jobs on this machine, `horizon=64`, `replay_ratio=1`, and `hidden=128` is now the fastest measured baseline at `510.8K` train SPS. Hidden size `256` is still too expensive for this stage. Thread count is not monotonic; 4 threads slightly beat 8/12 in this short run for the h32 variant, so future sweeps should compare quality as well as throughput before changing the default.

## Residual Controller Throughput

The local PyTorch PPO throughput benchmark now supports controller mode, residual scale, and task-conditioned observations so it can measure the same path used by the teacher-residual circle checkpoints.

Residual controller command:

```bash
python scripts/benchmark_sixdof_training_throughput.py \
  --variants smoke_64x16_h64 base_256x32_h128 \
  --output artifacts/replay/sixdof_residual_training_throughput_2026-05-20.json \
  --task circle \
  --controller teacher_residual \
  --residual-scale 0.05 \
  --native-step
```

| variant | envs | horizon | hidden | collect SPS | update SPS | total SPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke_64x16_h64 | 64 | 16 | 64 | 225005 | 223571 | 112143 |
| base_256x32_h128 | 256 | 32 | 128 | 475410 | 372941 | 208994 |

Standalone policy comparison:

```bash
python scripts/benchmark_sixdof_training_throughput.py \
  --variants smoke_64x16_h64 base_256x32_h128 \
  --output artifacts/replay/sixdof_policy_training_throughput_2026-05-20.json \
  --task circle \
  --controller policy \
  --native-step
```

| variant | envs | horizon | hidden | collect SPS | update SPS | total SPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke_64x16_h64 | 64 | 16 | 64 | 218617 | 231062 | 112334 |
| base_256x32_h128 | 256 | 32 | 128 | 529822 | 370099 | 217893 |

Result: teacher-residual rollout collection is slightly slower than direct policy rollout in this short local PyTorch benchmark, but it stays in the same throughput class. The stable residual path is therefore practical enough for the next residual-quality sweeps; the native/Puffer runner remains the faster path for large CPU sweeps.
