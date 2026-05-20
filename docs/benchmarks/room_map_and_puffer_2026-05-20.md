# Room Map And Puffer Checks

Date: 2026-05-20

Scope: unattended, sim/log-only validation. No live Crazyflie hardware commands were run.

## Room Map Quality

Command:

```bash
python scripts/summarize_crazyflie_room.py \
  --input artifacts/crazyflie_logs/room_scan_autonomous_35s.csv \
  --output artifacts/replay/room_scan_autonomous_35s.room.json \
  --markdown artifacts/replay/room_scan_autonomous_35s.room.md
```

Result: the autonomous room scan is mapping-ready under the default gate.

| metric | value |
| --- | ---: |
| points | 7416 |
| poses | 2098 |
| duration s | 34.95 |
| trajectory xy span m | 1.684 |
| trajectory path length m | 4.469 |
| point cloud xy span m | 5.059 |
| point cloud z span m | 2.435 |

All four horizontal Multi-ranger directions were active. `range.zrange` contributed no accepted points in this log, so floor range should still be checked in a dedicated hover log before using the point cloud for vertical calibration.

The abort log `artifacts/crazyflie_logs/room_scan_airborne_40s.csv` is not mapping-ready: it has only 2 accepted points, 1 pose sample, no duration, and only back/right horizontal coverage.

## Estimated Room Bounds

`scripts/summarize_crazyflie_room.py` now writes an axis-aligned room estimate for sim-side replay checks. Current bounds from `room_scan_autonomous_35s.csv`:

| bound | value m |
| --- | ---: |
| x_min | -1.751 |
| x_max | 0.624 |
| y_min | -1.756 |
| y_max | 1.682 |
| z_min | 0.000 |
| z_max | 2.415 |
| width | 2.375 |
| depth | 3.438 |
| height | 2.415 |

Warning: `floor_from_default`. The scan did not provide accepted down-ranger floor points, so the estimate uses `z_min=0.0`.

Smoke rollout using this room estimate:

```bash
python scripts/rollout_sixdof_policy.py \
  --teacher \
  --task obstacle_avoidance \
  --room-report artifacts/replay/room_scan_autonomous_35s.room.json \
  --steps 120 \
  --seed 51 \
  --output artifacts/trajectories/sixdof_teacher_room_estimate_smoke.csv
```

Result: 120 simulated rows. Time-aligned replay comparison against the real room scan currently overlaps only 1.19 seconds because this is not a command-matched replay, but it proves the measured-room bounds can drive the Python 6-DoF sim path. Custom room reports are intentionally rejected with `--native-step` until the native C kernel accepts configurable room bounds.

## Native 6-DoF Throughput

Command:

```bash
python scripts/benchmark_sixdof_sweep.py \
  --env-counts 1024 4096 8192 \
  --steps 2000 \
  --output artifacts/replay/sixdof_native_benchmark_latest.json
```

Best native env throughput: 16,950,295 steps/sec at 4096 envs. This confirms the C/native stepping path is not the immediate bottleneck for short CPU training runs.

## PufferLib Sweep

Command:

```bash
python scripts/run_sixdof_puffer_sweep.py \
  --run \
  --max-variants 5 \
  --total-timesteps 524288 \
  --output artifacts/replay/sixdof_puffer_sweep_latest.json
```

| variant | agents | threads | horizon | replay | hidden | train SPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small_h16_rr2_h64 | 1024 | 1 | 16 | 2 | 64 | 393200 |
| base_h32_rr2_h128 | 4096 | 8 | 32 | 2 | 128 | 263300 |
| fast_h32_rr1_h128 | 4096 | 8 | 32 | 1 | 128 | 475800 |
| wide_h32_rr1_h256 | 4096 | 8 | 32 | 1 | 256 | 271300 |
| large_h32_rr1_h128 | 8192 | 8 | 32 | 1 | 128 | 472400 |

Current fastest short-run Puffer setting: `fast_h32_rr1_h128`, narrowly ahead of `large_h32_rr1_h128`. Wider hidden size 256 roughly halves train SPS in this setup.
