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
