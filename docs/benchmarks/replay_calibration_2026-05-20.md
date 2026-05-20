# Replay Calibration Baseline

Date: 2026-05-20

Command:

```bash
python scripts/fit_replay_calibration.py \
  --real artifacts/crazyflie_logs/avoidance_policy.csv \
  --sim artifacts/trajectories/sixdof_obstacle_avoidance_h256_long.csv \
  --output artifacts/replay/avoidance_policy_vs_sixdof_obstacle_calibration.json
```

Model: `real ~= scale * sim + bias`

| signal | samples | scale | bias | raw RMSE | fitted RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| range.front | 292 | 0.0354301 | 1631.3 | 1002.85 | 34.5221 |
| range.back | 4 | -3.25308e-14 | 3359 | 1989.07 | 9.09495e-13 |
| range.left | 320 | -0.0536246 | 1491.31 | 1236.55 | 43.8542 |
| range.right | 310 | 0.386368 | -136.247 | 1140.61 | 292.146 |
| range.up | 312 | 0.133929 | 1727.37 | 176.211 | 41.0312 |
| range.zrange | 319 | 0.096574 | 298.843 | 333.095 | 37.9632 |

Interpretation:

- This proves the fitter works, not that these parameters are physically meaningful.
- The source real log and sim rollout are not matched by initial state, room geometry, or command sequence.
- A useful calibration run should record a deliberate mapping flight with estimator pose, command setpoints, and a known room origin.
