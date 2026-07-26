# Room Scene Autonomy Ideas

This note captures a possible direction, not an implemented feature.

The current Crazyflie mapping stack can already turn Flow deck pose estimates
and Multiranger readings into sparse room points. That makes it plausible to
promote room scans into reusable simulation scenes: coarse bounds first,
Manhattan wall fits next, and explicit obstacles or target regions later.

## Existing Pieces

- Hardware logs include `stateEstimate.x/y/z`, attitude, and the six ranger
  directions: front, back, left, right, up, and down.
- `flightrl.hardware.ranger_map` projects range readings into point clouds and
  room bounds.
- `flightrl.hardware.manhattan_map` can post-process sparse wall hits into a
  coarse rectangular room fit.
- `SixDofCrazyflieEnv` already accepts a `BoxRoom` and raycasts the same ranger
  directions against the room.
- `scripts/rollout_sixdof_policy.py` can load a room report into the 6-DoF sim.
- Target vectors and local hold/target-direction policies already exist as
  training and hardware-control surfaces.

## Scene Artifact

A useful next abstraction would be a small `RoomScene` JSON artifact generated
from scan outputs:

```json
{
  "version": 1,
  "frame": "scan_local",
  "bounds": {
    "x_min": -2.0,
    "x_max": 2.0,
    "y_min": -1.5,
    "y_max": 1.5,
    "z_min": 0.0,
    "z_max": 2.4,
    "max_range_m": 4.0
  },
  "obstacles": [
    {"id": "table", "type": "axis_aligned_box", "x_min": 0.4, "x_max": 0.9, "y_min": -0.3, "y_max": 0.4, "z_min": 0.0, "z_max": 0.8}
  ],
  "targets": [
    {"id": "center_hover", "position_m": [0.0, 0.0, 0.55]},
    {"id": "far_corner", "position_m": [1.4, 1.0, 0.55]}
  ],
  "paths": [
    {"id": "inspection_loop", "targets": ["center_hover", "far_corner"]}
  ]
}
```

The first version should support bounds and named target points only. Obstacles,
paths, and semantic labels can be added once the room-frame quality is reliable.

## Training Uses

Room scenes would let the simulator train policies against spaces that resemble
the real test room:

- reach a named target while preserving clearance;
- follow a target sequence or inspection loop;
- recover from blocked starts near a wall;
- track a moving target or moving waypoint;
- scan/cover a room with repeated viewpoints;
- transfer a chosen room target to a conservative live setpoint controller.

The single-drone version can reuse the existing `BoxRoom`, ranger observation,
target-position observation, and position/yaw reward surfaces.

## Swarm Direction

Swarm training is a separate environment layer. `num_envs` is vectorized
parallel simulation, not multiple drones in one room. A real swarm scene would
need:

- multiple agents in one shared `RoomScene`;
- per-agent pose, velocity, target, and ranger observations;
- inter-agent distance or raycast features;
- collision penalties and minimum separation rewards;
- optional shared policy weights with per-agent target assignments;
- mission-level tasks such as coverage, formation movement, and target handoff.

This should start in Python for correctness. Native/Ocean support can follow
once the scene and reward design are stable.

## Boundaries

Multiranger scans are sparse. They are enough for coarse room bounds, wall
structure, avoidance, and target-region planning, but not dense furniture-scale
3D reconstruction. The main risk is localization drift: target points are only
safe if the scan-local frame and live state estimate stay aligned.

The safe progression is:

1. room scan to `RoomScene`;
2. replay and visualize simulated rollouts inside that scene;
3. train target/path policies in sim;
4. shadow live target commands against real ranger logs;
5. run short confirmed live flights with a conservative controller and hard
   clearance aborts.
