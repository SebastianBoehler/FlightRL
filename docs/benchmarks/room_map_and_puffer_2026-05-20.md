# Room Map And Puffer Checks

Date: 2026-05-20

Scope: unattended, sim/log-only validation. No live Crazyflie hardware commands were run.

## Room Map Quality

```bash
python scripts/summarize_crazyflie_room.py \
  --input artifacts/crazyflie_logs/room_scan_autonomous_35s.csv \
  --output artifacts/replay/room_scan_autonomous_35s.room.json \
  --markdown artifacts/replay/room_scan_autonomous_35s.room.md \
  --min-yaw-span-deg 45
```

Result: the autonomous room scan is mapping-ready with the yaw-span gate.

| metric | value |
| --- | ---: |
| points | 7416 |
| poses | 2098 |
| duration s | 34.95 |
| trajectory xy span m | 1.684 |
| trajectory path length m | 4.469 |
| point density per path m | 1659.5 |
| trajectory yaw span deg | 556.2 |
| trajectory p95 speed m/s | 9.502 |
| trajectory max step speed m/s | 444.668 |
| point cloud xy span m | 5.059 |
| point cloud z span m | 2.435 |

All four horizontal Multi-ranger directions were active. `range.zrange` contributed no accepted points in this log, so floor range should still be checked in a dedicated hover log before using the point cloud for vertical calibration.

The new path-quality metrics show enough yaw coverage for a room scan, but also expose an estimator spike in `max_step_speed_m_s`. Prefer p95 speed for scan-quality judgement and inspect max-step spikes before using the trajectory for precise command-matched replay.

```bash
python scripts/summarize_crazyflie_room.py \
  --input artifacts/crazyflie_logs/room_scan_autonomous_35s.csv \
  --output artifacts/replay/room_scan_autonomous_35s.strict_path.room.json \
  --markdown artifacts/replay/room_scan_autonomous_35s.strict_path.room.md \
  --min-yaw-span-deg 45 \
  --max-step-speed-m-s 20
```

Result: strict path quality is not ready because of `speed_glitch`. The scan has `9` speed spikes above `20m/s` (`0.0043` of trajectory steps), with p95 speed `9.502m/s` and max step speed `444.668m/s`. This is still useful for coarse room bounds but should not be used as a command-matched replay log until the estimator spike is filtered or a cleaner scan is recorded.

The abort log `artifacts/crazyflie_logs/room_scan_airborne_40s.csv` is not mapping-ready: it has only 2 accepted points, 1 pose sample, no duration, and only back/right horizontal coverage.

Added a separate cleaner for replay-prep experiments:

```bash
python scripts/clean_crazyflie_room_log.py \
  --input artifacts/crazyflie_logs/room_scan_autonomous_35s.csv \
  --output artifacts/crazyflie_logs/room_scan_autonomous_35s.clean20.csv \
  --report artifacts/replay/room_scan_autonomous_35s.clean20.json \
  --max-step-speed-m-s 20
```

This keeps the raw log unchanged and writes a filtered CSV plus a drop-count report. The filtered log should be summarized with the same strict `--max-step-speed-m-s` gate before it is used as replay evidence.

Result at `20m/s`: the cleaner dropped 10 of 2098 rows (`0.0048`). The strict summary for `room_scan_autonomous_35s.clean20.csv` is now `mapping_ready=True` with zero remaining speed glitches, p95 speed `9.456m/s`, max step speed `18.876m/s`, 7385 accepted points, and the same coarse room bounds as the raw scan.

Readiness smoke with the cleaned room report:

```bash
python scripts/build_sixdof_readiness_report.py \
  --matrix artifacts/replay/sixdof_candidate_matrix_current.json \
  --room-report artifacts/replay/room_scan_autonomous_35s.clean20.strict_path.room.json \
  --native-parity artifacts/replay/sixdof_native_parity_current.json \
  --replay-comparison artifacts/replay/sixdof_self_replay_compare_smoke.json \
  --require-replay-comparison \
  --output artifacts/replay/sixdof_readiness_clean_room_replay_smoke.json
```

Result: obstacle avoidance is readiness-ready for sim/edge promotion; `position_yaw` and `multitask` remain blocked by `sim_gate`. This is still not live-flight approval.

Added command-matched replay scaffolding for real room-scan logs:

```bash
python scripts/replay_crazyflie_commands.py \
  --input artifacts/crazyflie_logs/room_scan_autonomous_35s.clean20.csv \
  --room-report artifacts/replay/room_scan_autonomous_35s.clean20.strict_path.room.json \
  --output artifacts/trajectories/room_scan_autonomous_35s.command_replay_zhold055.csv \
  --normalized-real-output artifacts/trajectories/room_scan_autonomous_35s.clean20.normalized_zhold055.csv \
  --velocity-gain 2.5 \
  --max-dt-s 0.08 \
  --override-z-m 0.55 \
  --hold-z-m 0.55
```

`--override-z-m` is explicit because this log reports `stateEstimate.z` near zero and has no accepted `range.zrange` floor hits. `--hold-z-m` avoids replaying the bad vertical command generated from that same unreliable floor range, so the comparison is a horizontal/yaw/ranger replay at the scan height, not a claim that vertical telemetry matched.

Comparison command:

```bash
python scripts/compare_crazyflie_replay.py \
  --real artifacts/trajectories/room_scan_autonomous_35s.clean20.normalized_zhold055.csv \
  --sim artifacts/trajectories/room_scan_autonomous_35s.command_replay_zhold055.csv \
  --align-time \
  --signals stateEstimate.x stateEstimate.y stateEstimate.z stateEstimate.vx stateEstimate.vy stateEstimate.vz stabilizer.yaw range.front range.back range.left range.right range.up \
  --output artifacts/replay/room_scan_autonomous_35s.command_replay_zhold055_compare.json
```

Result: the command replay aligns for the full `34.95s` and 2088 samples, but it is not yet replay-ready. Worst state RMSE is `0.575m` and worst ranger RMSE is `1331mm`; readiness with this comparison correctly blocks all candidates on `replay_comparison`. The comparator now wraps `stabilizer.yaw` as a circular degree signal, reducing yaw RMSE from the misleading unwrapped value to `52.98deg`.

Added a calibration sweep for the high-level command replay bridge:

```bash
python scripts/sweep_crazyflie_command_replay.py \
  --input artifacts/crazyflie_logs/room_scan_autonomous_35s.clean20.csv \
  --room-report artifacts/replay/room_scan_autonomous_35s.clean20.strict_path.room.json \
  --output artifacts/replay/room_scan_autonomous_35s.command_replay_sweep.json \
  --best-sim-output artifacts/trajectories/room_scan_autonomous_35s.command_replay_best.csv \
  --best-real-output artifacts/trajectories/room_scan_autonomous_35s.clean20.normalized_best.csv \
  --override-z-m 0.55 \
  --hold-z-values 0.55 \
  --velocity-gains 0.75 1.25 2.0 2.5 3.5 5.0 \
  --yawrate-scales 0.5 0.75 1.0 1.25 1.5 2.0 2.5 \
  --max-dt-values 0.04 0.05 0.08
```

Best current bridge parameters are `velocity_gain=0.75`, `yawrate_scale=1.25`, `max_dt_s=0.05`, `override_z_m=0.55`, and `hold_z_m=0.55`. This improves yaw RMSE to `14.84deg`, but the replay is still not calibration-ready: worst horizontal state RMSE is `0.588m`, and worst ranger RMSE is `1198mm`. The best-parameter readiness report at `artifacts/replay/sixdof_readiness_command_replay_best.md` still blocks all candidates on `replay_comparison`, which is the correct safety outcome.

Expanded the command replay sweep to test frame and sign conventions:

```bash
python scripts/sweep_crazyflie_command_replay.py \
  --input artifacts/crazyflie_logs/room_scan_autonomous_35s.clean20.csv \
  --room-report artifacts/replay/room_scan_autonomous_35s.clean20.strict_path.room.json \
  --output artifacts/replay/room_scan_autonomous_35s.command_replay_frame_sweep.json \
  --best-sim-output artifacts/trajectories/room_scan_autonomous_35s.command_replay_frame_best.csv \
  --best-real-output artifacts/trajectories/room_scan_autonomous_35s.clean20.normalized_frame_best.csv \
  --override-z-m 0.55 \
  --hold-z-values 0.55 \
  --velocity-gains 0.5 0.75 1.25 2.5 \
  --yawrate-scales 1.0 1.25 1.5 \
  --max-dt-values 0.05 \
  --command-frames body world \
  --yaw-sources logged sim \
  --vx-signs -1 1 \
  --vy-signs -1 1
```

Result: the best frame/sign candidate is still the original convention: `command_frame=body`, `yaw_source=logged`, `vx_sign=1`, `vy_sign=1`. The sweep slightly improves worst ranger RMSE to `1193mm` with `velocity_gain=0.5`, but worsens horizontal state RMSE to `0.597m`. This rules out a simple coordinate sign/frame bug as the dominant replay error. The next replay-fidelity milestone should use a cleaner calibration flight with trustworthy floor range/height and a known commanded trajectory instead of this exploratory scan.

Added a dedicated calibration-flight logger for that next milestone:

```bash
python scripts/crazyflie_calibration_flight.py \
  --confirm-flight \
  --output artifacts/crazyflie_logs/calibration_line_yaw_square.csv \
  --height-m 0.55 \
  --segment-s 1.6 \
  --hover-s 1.0 \
  --speed-m-s 0.12 \
  --yawrate-deg-s 20
```

Dry-run sequence duration is `18.00s`: hover, `line_x_pos`, `line_x_neg`, `line_y_pos`, `line_y_neg`, `yaw_pos`, `yaw_neg`, a four-side square, then hover. It logs ranger, pose, velocity, gyro, battery, and the exact commanded `vx_m_s`, `vy_m_s`, `vz_m_s`, `yawrate_deg_s`, and `mode` for replay fitting.

Calibration quality gate:

```bash
python scripts/summarize_crazyflie_calibration.py \
  --input artifacts/crazyflie_logs/calibration_line_yaw_square.csv \
  --output artifacts/replay/calibration_line_yaw_square.quality.json \
  --min-duration-s 8 \
  --min-rows 100 \
  --min-floor-valid-ratio 0.5 \
  --min-yaw-span-deg 45 \
  --strict
```

This should be run before using a real calibration flight for replay readiness. It rejects logs that do not include enough rows/duration, valid floor range, enough yaw motion, or both positive and negative x/y/yaw command modes.

Added a one-command replay evidence pipeline for a completed calibration log:

```bash
python scripts/build_calibration_replay_report.py \
  --input artifacts/crazyflie_logs/calibration_line_yaw_square.csv \
  --room-report artifacts/replay/room_scan_autonomous_35s.clean20.strict_path.room.json \
  --matrix artifacts/replay/sixdof_candidate_matrix_current.json \
  --native-parity artifacts/replay/sixdof_native_parity_current.json \
  --output-dir artifacts/replay/calibration_line_yaw_square \
  --override-z-m 0.55 \
  --hold-z-values 0.55 \
  --velocity-gains 0.5 0.75 1.25 2.5 \
  --yawrate-scales 1.0 1.25 1.5 \
  --max-dt-values 0.05 \
  --command-frames body world \
  --yaw-sources logged sim \
  --vx-signs -1 1 \
  --vy-signs -1 1
```

The pipeline writes quality, sweep, best replay CSVs, aligned comparison, and readiness artifacts. It refuses to run the sweep/readiness stages if the calibration quality gate fails unless `--allow-unready-quality` is supplied.

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

Result: 120 simulated rows. Time-aligned replay comparison against the real room scan currently overlaps only 1.19 seconds because this is not a command-matched replay, but it proves the measured-room bounds can drive the 6-DoF sim path.

Native-step parity command:

```bash
python scripts/rollout_sixdof_policy.py \
  --teacher \
  --native-step \
  --task obstacle_avoidance \
  --room-report artifacts/replay/room_scan_autonomous_35s.room.json \
  --steps 120 \
  --seed 51 \
  --output artifacts/trajectories/sixdof_teacher_room_estimate_native_smoke.csv
```

Python-vs-native replay parity over the measured room produced 120 aligned samples. State RMSE was below `4e-8` m and ranger RMSE was below `6e-4` mm-equivalent CSV units, so configurable room bounds now match across Python and native stepping for this smoke trajectory.

## Native 6-DoF Throughput

```bash
python scripts/benchmark_sixdof_sweep.py \
  --env-counts 1024 4096 8192 \
  --steps 2000 \
  --output artifacts/replay/sixdof_native_benchmark_latest.json
```

Best native env throughput after configurable room bounds: 15,984,486 steps/sec at 4096 envs. This confirms the C/native stepping path is still not the immediate bottleneck for short CPU training runs.

## PufferLib Sweep

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

## PPO Training Throughput

Native env stepping is not the full training bottleneck, so `scripts/benchmark_sixdof_training_throughput.py` measures rollout collection plus one PPO update over common local configurations.

```bash
python scripts/benchmark_sixdof_training_throughput.py \
  --output artifacts/replay/sixdof_training_throughput_latest.json \
  --native-step
```

| variant | collect SPS | update SPS | total SPS |
| --- | ---: | ---: | ---: |
| smoke_64x16_h64 | 328302 | 306748 | 158579 |
| base_256x32_h128 | 647316 | 421401 | 255240 |
| wide_256x32_h256 | 288457 | 300941 | 147283 |
| large_512x32_h128 | 447155 | 591789 | 254702 |
| long_256x64_h128 | 479504 | 623918 | 271130 |

Current fastest end-to-end PPO local setting: `long_256x64_h128` at `271130` samples/sec. Best collection-only setting: `base_256x32_h128` at `647316` samples/sec. Hidden size 256 again costs about half the end-to-end throughput, so 128 remains the practical default for sweep breadth.
