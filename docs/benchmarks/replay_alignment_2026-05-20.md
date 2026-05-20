# Replay Alignment Baseline

Date: 2026-05-20

Command:

```bash
python scripts/compare_crazyflie_replay.py \
  --real artifacts/crazyflie_logs/avoidance_policy.csv \
  --sim artifacts/trajectories/sixdof_obstacle_avoidance_h256_long.csv \
  --align-time \
  --output artifacts/replay/avoidance_policy_vs_sixdof_obstacle_aligned.json
```

Aligned overlap: `7.99` seconds, `320` real timestamps.

The real log contains range and command telemetry but no `stateEstimate.*` position columns, so this baseline compares ranger signals only. Invalid range sentinels such as `32766` are excluded from aligned range metrics.

| signal | samples | RMSE | bias | real mean | sim mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| range.back | 4 | 1989.066 | -1989.064 | 3359.000 | 1369.936 |
| range.front | 292 | 1002.848 | 925.894 | 1725.233 | 2651.126 |
| range.left | 320 | 1236.550 | 1227.529 | 1352.938 | 2580.467 |
| range.right | 310 | 1140.609 | 1088.570 | 463.374 | 1551.944 |
| range.up | 312 | 176.211 | -150.986 | 1971.141 | 1820.155 |
| range.zrange | 319 | 333.095 | 320.768 | 365.078 | 685.847 |

Interpretation:

- This is a signal-shape comparison, not an exact replay. The real policy and sim rollout do not share initial state, command source, or room geometry.
- The large horizontal range biases are expected under that mismatch and are useful as a calibration target.
- A stronger replay gate needs real logs that include `stateEstimate.x/y/z`, command setpoints, room-start reset, and a matching simulated command sequence.

## Readiness Gate Wiring

`scripts/build_sixdof_readiness_report.py` now accepts `--replay-comparison` plus `--require-replay-comparison`. When a `compare_crazyflie_replay.py --align-time` JSON report is supplied, readiness checks overlap duration, worst `stateEstimate.*` RMSE, and worst `range.*` RMSE before promoting a candidate.

Smoke command using a self-comparison to validate the gate path without live hardware:

```bash
python scripts/compare_crazyflie_replay.py \
  --real artifacts/trajectories/sixdof_teacher_room_estimate_native_smoke.csv \
  --sim artifacts/trajectories/sixdof_teacher_room_estimate_native_smoke.csv \
  --align-time \
  --output artifacts/replay/sixdof_self_replay_compare_smoke.json

python scripts/build_sixdof_readiness_report.py \
  --matrix artifacts/replay/sixdof_candidate_matrix_current.json \
  --room-report artifacts/replay/room_scan_autonomous_35s.room.json \
  --native-parity artifacts/replay/sixdof_native_parity_current.json \
  --replay-comparison artifacts/replay/sixdof_self_replay_compare_smoke.json \
  --require-replay-comparison \
  --output artifacts/replay/sixdof_readiness_with_replay_smoke.json
```

Smoke result: replay comparison passed with `120` aligned samples, `1.19s` overlap, worst state RMSE `0.0`, and worst range RMSE `0.0mm`. This proves the readiness gate can consume replay evidence; it is not real-flight replay evidence because both sides are the same simulated CSV.
