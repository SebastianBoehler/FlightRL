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
