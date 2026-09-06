# Camera-based control experiment

A new shared actor controls three FPV-reference aircraft using actual native RGB-D
camera observations. It receives no world positions, target coordinates, planned
routes, collision geometry, simulator visibility counts or elapsed mission phase.

## Inputs, actions and coordination

The actor consumes RGB (64 × 48), ideal simulated depth, ideal body velocity and
attitude estimates, gyro, a one-hot role, and delayed peer detection reports with
validity and age fields. The simulator generates those measurements from its state;
this experiment does not validate a real visual-inertial estimator or sensor noise.

One set of weights produces collective thrust, roll/pitch/yaw-rate commands and a
visual-report confidence. Three consecutive confidence samples above 0.8 trigger a
report. The red and blue scouts approach their beacons; the green confirmer receives
their reports after 200 ms. There is no route planner or classical flight-controller
correction at actor inference. Native vehicle rate response and thrust lag remain
part of the simulated dynamics. Body-rate commands are not individual motor outputs.

The teacher used for imitation derives visual-servo commands from image colour and
depth. It does not use target coordinates. Initial training used randomized renders;
DAgger then added the actual images encountered by teacher and learner rollouts.
This is imitation learning, not reinforcement learning. The markers are synthetic
colour beacons, not people or learned semantic object categories.

## Evaluation provenance

`artifacts/camera-control-20260906` preserves the initial training, all three DAgger
rounds and the first frozen test set, seeds 3300–3311. The initial model collided in
all three development runs. After DAgger, nominal evaluation completed 12/12 cases.
Removing RGB-D caused 12 collisions. Withholding messages also caused 12 collisions:
that failed ablation exposed missing-message distribution shift, not robust teamwork.

`artifacts/camera-control-linkloss-20260906` is a separate revision trained with
missing-message examples from seeds 3168–3179. It reserves new test seeds 3400–3411.
The original evaluation is retained and is not reused for tuning this revision.
Run `scripts/evaluate_camera_control.py <artifact-directory>` for a frozen actor;
the evaluator raises if the classical teacher is called or an evaluation is replaced.

The dashboard displays a predecoded atlas of exact recorded actor RGB frames at
10 Hz, aligned with the replay index. The 3-D overview is observer-only. Those are
actual policy inputs, rather than high-detail images rendered after the run.

## Limits and broader robot pipeline

This is a small indoor mission with two analytic obstacles and three visible colour
beacons. Seeds vary initial poses and beacon heights on the same layout. It does not
establish open-world obstacle discovery, forest search, semantic recognition, LiDAR
control, unfamiliar vehicle transfer, severe camera occlusion, wind robustness or
safe operation in arbitrary environments. Existing forest/SAR recordings remain
separate policies; they have not silently become camera-only.

The actor is bound to the existing versioned sensor/action contract with explicit
shapes, units, body/camera frames and native actuator order. The reusable boundary
is robot measurements → policy → vehicle-specific actuator adapter. Supporting
another robot or sensor requires an implemented simulator/sensor adapter, a matching
policy contract and training/evaluation; importing arbitrary geometry alone does
not supply its dynamics or a trained policy. LiDAR and other modalities can fit that
contract, but this camera actor does not consume them.

## Revised frozen results

All 12 new nominal cases completed without reported collisions. With peer messages
withheld, both scouts reported and the confirmer remained unconfirmed in all 12
cases, ending at the 49.9-second time limit without reported collisions. Removing
RGB-D still caused collisions in all 12 cases: blind-flight recovery is not solved.

Nominal duration: 19.7–20.4 seconds. Minimum peer surface clearance: 1.51 m. Measured on an Apple M4 Max with two PyTorch CPU threads, end-to-end throughput: 2268 aircraft camera/control steps per second over 12.6 seconds, including native RGB-D rendering, actor inference, physics, collision checks and recording. Rendering the dashboard and gradient updates are excluded.
