# Handheld Crazyflie Room Map

This was a passive hardware capture: no arming, no motor commands, no takeoff. The drone was powered and hand-carried through the room while logging Flow/state-estimate, attitude, battery, and all Multiranger directions.

## Capture

Discovery/check:

```bash
python scripts/crazyflie_bringup.py scan
python scripts/crazyflie_bringup.py check
```

Result:

- URI found: `radio://0/80/2M`
- Device: `Crazyflie 2.1 Brushless`
- Firmware: `2024.10.2`
- Flow deck: detected
- Multiranger deck: detected
- Log variables: `22`

Passive log:

```bash
python scripts/crazyflie_log.py \
  --duration-s 45 \
  --output artifacts/crazyflie_logs/handheld_room_map_sync_2026-05-20.csv
```

The passive logger now merges cflib log blocks before writing rows, so each CSV row has a complete pose/ranger snapshot instead of partial block rows.

## Map Artifacts

- Raw synced log: `artifacts/crazyflie_logs/handheld_room_map_sync_2026-05-20.csv`
- Cleaned log: `artifacts/crazyflie_logs/handheld_room_map_sync_2026-05-20.clean3.csv`
- Clean point cloud: `artifacts/crazyflie_logs/handheld_room_map_sync_2026-05-20.clean3.room.png`
- Quality report: `artifacts/crazyflie_logs/handheld_room_map_sync_2026-05-20.clean3.room.json`

Cleaning used a `3.0 m/s` max step-speed filter and dropped `1303` of `4492` rows. The raw hand-carried estimator had speed spikes up to `362 m/s`, so the cleaned map is the useful artifact.

## Clean Map Quality

| metric | value |
| --- | ---: |
| mapping_ready | true |
| point_count | 16330 |
| pose_count | 3189 |
| duration_s | 44.94 |
| active_horizontal_sensors | 4 |
| trajectory_xy_span_m | 7.398 |
| trajectory_path_length_m | 26.051 |
| trajectory_yaw_span_deg | 240.0 |
| point_cloud_xy_span_m | 10.768 |
| point_cloud_z_span_m | 2.620 |

Estimated room box:

| bound | value m |
| --- | ---: |
| x_min | -2.291 |
| x_max | 3.639 |
| y_min | -1.064 |
| y_max | 5.216 |
| z_min | -0.079 |
| z_max | 2.400 |
| width | 5.930 |
| depth | 6.280 |
| height | 2.479 |

Conclusion: the passive hand-carried workflow works well enough for room-map data collection. The next mapping improvement is estimator smoothing or trajectory post-processing, because hand-carrying produces occasional Flow/state-estimate jumps even though ranger coverage is strong.
