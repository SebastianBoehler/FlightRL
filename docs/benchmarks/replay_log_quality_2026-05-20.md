# Replay Log Quality Baseline

Date: 2026-05-20

Command template:

```bash
python scripts/check_replay_log_quality.py \
  --input artifacts/crazyflie_logs/ranger_hold_current_target_35s.csv \
  --output artifacts/replay/ranger_hold_current_target_quality.json
```

Calibration-ready logs must include host time, estimator position, velocity/yaw command setpoints, and front/back/left/right/up ranger columns. The default gate also requires at least 100 rows, at least 5 seconds duration, strictly increasing timestamps, and at least 25 percent valid samples for required ranger columns.

| log | ready | rows | duration s | sample rate Hz | failures |
| --- | ---: | ---: | ---: | ---: | --- |
| ranger_hold_current_target_35s.csv | true | 2794 | 34.928 | 79.965 | none |
| reactive_avoidance_visualized_30s.csv | true | 1798 | 29.944 | 60.011 | none |
| avoidance_policy.csv | false | 1199 | 29.945 | 40.006 | missing_columns |

`avoidance_policy.csv` has useful ranger/command telemetry but lacks `stateEstimate.x/y/z`, so it is not suitable for physical replay calibration. Prefer the two ready logs for the next matched replay and room-calibration passes.
