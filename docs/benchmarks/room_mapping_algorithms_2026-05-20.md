# Crazyflie Room Mapping Algorithms

The current HTML point cloud is a direct projection of sparse range hits into the Crazyflie state-estimate frame. That is useful for debugging sensor coverage, but it is not a full SLAM map. The Multiranger has five single-point ToF sensors, not a dense lidar or depth camera, so map quality depends heavily on pose accuracy and scan coverage.

## Findings

- Bitcraze documents the Multiranger as nearest-surface ranging in five directions: front, back, left, right, and up, with a nominal range up to 4 m. It explicitly provides sensor data only; obstacle reaction and mapping must be implemented in software.
- Bitcraze's ROS 2 mapping tutorial routes Crazyflie Flow/Multiranger data into ROS topics and uses ROS tooling for mapping. Their experimental `crazyflie_ros2_multiranger` package includes simple mapper, wall-following mapper, simulation, real-drone launch files, and RViz visualization.
- `slam_toolbox` is the established ROS 2 path for 2D lidar-style mapping, pose-graph optimization, map serialization, and continued mapping. It expects usable odometry plus laser-like scans.
- Sparse four-point lidar papers target exactly this problem: few single-direction range sensors plus noisy pose. The recurring solution is not raw point clouds, but Manhattan-world structure: convert range hits to points, fit wall lines/planes with RANSAC or axis alignment, and use the fitted walls to correct or stabilize the map.

## Local Implementation

Added a first Manhattan-world post-processor:

```bash
python scripts/fit_crazyflie_room_manhattan.py \
  --points artifacts/crazyflie_logs/handheld_room_scan_2026-05-20.room_points.csv \
  --output-prefix artifacts/crazyflie_logs/handheld_room_scan_2026-05-20 \
  --angle-samples 360 \
  --quantile 0.03 \
  --max-wall-residual-m 0.35
```

Generated artifacts:

- `artifacts/crazyflie_logs/handheld_room_scan_2026-05-20.manhattan.html`
- `artifacts/crazyflie_logs/handheld_room_scan_2026-05-20.manhattan.ply`
- `artifacts/crazyflie_logs/handheld_room_scan_2026-05-20.manhattan_points.csv`
- `artifacts/crazyflie_logs/handheld_room_scan_2026-05-20.manhattan.json`

Result on the handheld scan:

| metric | value |
| --- | ---: |
| horizontal source points | 9986 |
| snapped wall points | 2067 |
| wall fraction | 0.207 |
| median wall residual m | 0.724 |
| fitted angle deg | 85.75 |
| fitted width m | 5.78 |
| fitted depth m | 5.17 |

This is a diagnostic improvement, not a solved room reconstruction. The low wall fraction says the handheld capture still contains a lot of drift, close-range clutter, or non-wall hits. A better capture should fly or hand-carry slowly at a steady height, rotate deliberately near the center, avoid hands/people near sensors, and log enough repeated wall views for line fitting.

## Existing Log Comparison

Ran the same Manhattan analysis on the existing usable room/ranger logs:

| log | horizontal points | snapped points | wall fraction | median residual m | fitted width m | fitted depth m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `handheld_room_map_sync_2026-05-20` | 9986 | 2067 | 0.207 | 0.724 | 5.78 | 5.17 |
| `reactive_avoidance_visualized_30s` | 6049 | 2140 | 0.354 | 0.793 | 3.21 | 4.90 |
| `room_scan_autonomous_35s` | 5799 | 2282 | 0.394 | 0.473 | 3.00 | 2.02 |

`room_scan_airborne_40s` was skipped for Manhattan fitting because it only had 2 usable horizontal ranger points after export.

## Recommended Path

1. Keep the current direct point-cloud renderer as a raw sensor debugger.
2. Use the Manhattan fitter as the first room-shape view for sparse Multiranger scans.
3. Add a ROS 2 export/bridge next: publish `/scan`, `/odom`, and TF from our CSV/live logger so the same data can be tested with `slam_toolbox`.
4. For real 3D reconstruction, add a denser perception deck later: AI deck camera, external depth, Lighthouse/mocap, or a separate RGB-D capture source. Multiranger alone is enough for avoidance and coarse room geometry, not high-resolution room meshes.

## Sources

- Bitcraze Multiranger deck: https://www.bitcraze.io/products/multi-ranger-deck/
- Bitcraze ROS 2 mapping tutorial/tag: https://www.bitcraze.io/tag/mapping/
- Crazyflie ROS 2 Multiranger examples: https://github.com/knmcguire/crazyflie_ros2_multiranger
- ROS 2 `slam_toolbox`: https://github.com/SteveMacenski/slam_toolbox
- Four-point lidar Manhattan mapping paper: https://mpil-gist.github.io/assets/paper/2022_iccas_parsing.pdf
- Linear four-point lidar SLAM paper: https://mpil-gist.github.io/assets/paper/2023_ral_linear.pdf
