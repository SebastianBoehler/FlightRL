# Unified missions and learned 3-D flight control

The default dashboard now selects single-drone recordings, cooperative takeover,
forest search-and-confirm, utility-plant inspection, and learned 3-D flight control.

## Direct flight experiment

`artifacts/direct-flight-20260906` contains the frozen plan, checkpoint, training
metrics, complete evaluation outcomes and preselected seed-200 replay. Run
`scripts/train_direct_flight.py` in a fresh artifact directory to reproduce training;
`scripts/evaluate_direct_flight.py` evaluates the saved checkpoint.

The 10,948-parameter controller learned from 40,000 classical-controller state/action
examples, with 4,000 separate validation states. It consumes relative XYZ waypoint,
velocity, attitude and heading error. It directly outputs normalized collective
thrust and roll, pitch and yaw-rate commands. Native six-DOF integration advances at
50 Hz and policy inference at 10 Hz. This is behavioral cloning, not reinforcement
learning or individual-motor control. The native rate response remains part of the
simulated vehicle.

All 12 frozen test seeds completed: three parallel aircraft each traversed ten
prescribed waypoints, climbed over a 1.4 m barrier, descended below a 1.6 m overhead
obstacle, turned and returned. Altitude excursion was 2.24–2.35 m. No collision was
reported; minimum peer surface clearance was 1.71 m. The test seeds vary waypoint
altitudes on the same obstacle layout, not twelve independent environments. The
classical action-generation function was replaced with an exception during testing
to establish that learned inference did not silently use the teacher.

Measured locally: 7,460 joint three-aircraft control steps/s, equivalent to 111,903
individual aircraft physics substeps/s, over a short 0.79-second measurement.
This includes learned CPU inference, native dynamics, swept collision checks and
record construction. It excludes rendering, model loading and gradient updates;
it is a short throughput sample, not a sustained training benchmark.

These results establish bounded waypoint-conditioned flight, not safe navigation
in arbitrary environments. Obstacle-aware routing is prescribed here. Camera
perception, full scene collision meshes, terrain generalization, wind robustness,
vehicle transfer and learned mission-level planning remain unvalidated by this test.

## Mission catalog

`artifacts/mission-catalog-20260906` retains six new evaluation seeds (142–147)
for each mission. The existing learned task-bid model is reused without retraining.
Forest search-and-confirm and utility-plant inspection each completed 6/6 cases.
These missions still use the classical XYZ controller, separate from the direct
flight experiment above.

Two scouts inspect known sector centers and detect synthetic beacons using proximity
and geometric line of sight. A confirmation aircraft becomes eligible only after a
0.2-second scout report delay. Each of nine targets requires both detection and
confirmation before return. This is a role-coordination rehearsal, not visual person
recognition or open-world search. Small varying inspection heights replace completely
flat trajectories, while conservative altitude separation remains.

The utility mission checks assigned waypoints, not equipment fault recognition.
The original failed fleet pilot remains selectable, rather than being replaced by
successful demonstrations. The environment fidelity review is linked from the
shared dashboard header at `/review/fidelity.html`.
