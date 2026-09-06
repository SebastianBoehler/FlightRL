# Mixed-robot industrial inspection — 2026-09-06

First implemented milestone of the [robotics direction](research/robotics-platform-direction-20260906.md).
For the subsequent power/production sites, modeled sensor errors and shared learned
drone/rover controller, see [industrial expansion](industry-expansion-20260906.md).
A learned camera-driven drone and a classical camera-driven differential-drive
rover now inspect the same industrial corridor, exchange an inspection report,
and return the rover to a docking marker. This is a bounded simulator experiment.

## What runs

- `assets/robots/inspection_pair.xml`: authored surrogate MJCF models. The rover
  has driven wheel hinges, force limits and low-friction caster contacts. The
  drone has filtered collective/body-rate actuation with bounded torques.
- `src/flightrl/robotics/spec.py`: mechanical identity imported from the compiled
  model: link masses, COMs, principal inertias, joints, actuator gains/limits,
  sensors and camera mounts. Source hashes accompany each episode. This first
  importer supports the demonstrated MJCF joint-transmission path; it is not an
  arbitrary URDF/SDF importer or a validated commercial robot library.
- `world.py`: MuJoCo 3.10.0 owns both robots and all industrial contacts. The
  scene renderer consumes those body transforms and compiled collision shapes.
  Jolt remains the forest backend; it does not also integrate these robots.
- `sensing.py`: two actual WebGPU RGB-D camera streams, IMU/encoder telemetry,
  timestamped observations and AprilTag 36h11 detection. Sensor cameras exclude
  observer labels/trails and their own chassis.
- `mission.py`: elevated marker 17 inspection, ground marker 23 inspection,
  delayed drone report, and marker 42 docking. The rover waits if the report
  cannot arrive. A restored link delivers the queued report. This is a small
  explicit message protocol, not a radio propagation simulation.
- `policy.py`: a 613,989-parameter RGB-D drone actor trained by imitation. It
  receives 128×96 RGB, metric depth and nine proprioceptive channels, and emits
  body velocity/yaw setpoints plus inspection confidence. The classical drone
  baseline and rover controller use image detections and depth. Target positions
  are used to generate scenarios and independently score reports, never as actor
  inputs or teacher action labels.
- `viewer/robotics.html`: freely orbitable industrial view, colored direction
  guides/trails, both camera feeds, mission state, link interruption, pause/save
  and reset. WebGPU is required; renderer and connection failures are visible.

The learned actor is specialized to elevated marker 17 in this corridor. It is
not the old forest CTBR checkpoint, a learned rover controller or a general
navigation policy. The previous forest actor remains untrained for that forest.

## Clocks and physical scope

MuJoCo integrates at 500 Hz; controls update at 50 Hz; camera batches are requested
at 10 Hz. Each batch renders two 512×384 perception views and two 128×96 actor
views, each with RGB and metric ray-distance depth. Vertical FOV is 63 degrees,
range is capped at 8 m, and images use top-left pixel order. The observer is capped
at 30 fps and 1536×864 pixels while preserving its aspect ratio.

World coordinates are meters with Z up. Body axes are X forward, Y left, Z up;
render poses use xyzw quaternions, while imported MuJoCo inertia/mount metadata
explicitly uses wxyz. Rover controls are wheel rad/s, limited to ±6 rad/s and
±3 Nm. Wheel radius is 0.11 m and track is 0.42 m. Drone actuation metadata records
its force/rate/torque limits and actuator delays.

Proprioception currently assumes ideal estimated body velocity and gravity plus
simulated gyro signals. IMU/encoder readings are exposed, but a noisy onboard
state estimator is not implemented. Mechanics, motor dynamics and material
friction are surrogate values, not identified from a physical robot. The drone
is a compact rigid-body collision approximation. This milestone does not add
articulated arms, dogs, humanoids, marine dynamics, tire deformation or fluid CFD.

## Recorded evidence

Local artifacts: `artifacts/robotics-implementation-20260906/`.

| Check | Result | Artifact |
| --- | --- | --- |
| Full image-servo baseline | Three reports independently verified; 54.0 s | `baseline-v5/*/report.json` |
| Frozen learned actor, seeds 101/102/103 | 3/3 complete inspection and docking missions; 54.92/54.94/54.22 s | `held-out/suite.json` |
| Render/physics depth agreement | 336 rays across 7 actual captured frames; maximum 2.17 mm error | `depth-parity.json` |
| Validation reports | 12 true positives, 1 false positive, 5 false negatives at confidence >0.8 | `training/validation.json` |
| Validation command MAE | 0.012/0.019/0.018 m/s XYZ, 0.015 rad/s yaw | `training/validation.json` |

Dataset splits were fixed before training: 2,000 poses from seeds 11–14, 400
validation poses from seeds 15–16, then untouched closed-loop seeds 101–103.
These seeds vary inspection-marker X positions within ±0.2 m; the building layout
is shared. Training poses vary approach distance, lateral/vertical displacement,
attitude and velocities. Three successes do not establish generalization to
unseen buildings, weather, robot models or hardware.

Training used Apple MPS, 100 epochs, and selected epoch 37 using validation alone.
The frozen actor SHA-256 is
`f2b0afc35623b8be81dc31744fcb4100a88260d614228f7b852baa03c8dd76fb`.
Training data hashes, history and the model contract are recorded alongside it.
No held-out outcome was used to change the actor.

On this Apple M4 Max / 36 GB Mac, the three 830×601 observer runs delivered median
30.0 fps and 9.98–9.99 camera batches/s. End-to-end camera batch p95 was 31.3–49.1 ms
against a 100 ms period; physics p95 was 1.51–2.40 ms per 20 ms control update.
These are measured pipeline timings, not GPU-utilization percentages or a promise
that resolution/robot count can scale linearly. At 1536×864, a 120-second link-outage run delivered median 30.0 fps and
9.99 camera batches/s, with camera-batch p95 27.22 ms and physics p95 1.81 ms.
That run correctly timed out while the rover waited for the missing report; it
is a failure/recovery test, not an additional mission success. Final link tests
are recorded in `interactive-final/`. With the learned actor, a second run
interrupted the link before inspection, verified that the rover waited, then
restored it and completed all three checks in 74.02 s at the same 1536×864
resolution (`8f2f7939/report.json`).

All 25 focused checks passed. They cover wheel traction versus free-spinning wheels, wall contact,
shared ground contact, model identity, rotated camera mounts, marker occlusion,
wrong/missing marker rejection, communications recovery and timestep refinement.
The saved depth comparison includes float16 storage error and does not discard
raster-edge samples. Browser console checks and the viewer production build also
pass. The original forest was reopened after these shared-camera changes and
verified with three live RGB-D feeds, 30 fps / 10 Hz, and no console errors. The existing Three.js chunk-size build warning remains.

Earlier failures are retained: a distant marker was unreadable, an oversized
marker clipped the near camera, the return-side support occluded a marker, and
scoring initially compared camera range to chassis coordinates. The final model
uses readable marker dimensions, 512×384 perception, unobstructed supports and
camera-mount coordinates for independent scoring.

## Run from this checkout

Use the existing `.venv` with the `realism` and `mujoco` dependency extras, and
install viewer packages with `npm ci --prefix viewer` if needed. Keep the viewer
and one robotics bridge running in separate terminals:

```sh
npm run dev --prefix viewer -- --port 4173
PYTHONPATH=src .venv/bin/python scripts/run_robotics.py \
  --actor artifacts/robotics-implementation-20260906/training/actor.pt \
  --output artifacts/robotics-live
```

Open `http://127.0.0.1:4173/robotics.html`. Omit `--actor` for the explicit classical
baseline. Each connection writes a separate episode. Choose a new output root
when restarting the process; existing experiment outputs are not overwritten.
Reset mission starts a new episode. Hiding the page pauses the simulation.

To reproduce training, stop the robotics bridge first, then run
`scripts/collect_robotics.py --output <new-dataset>` with `PYTHONPATH=src` and the
venv Python. Open the robotics page once the collector reports ready and keep it
visible until it saves training and validation data. Then run
`scripts/train_robotics.py --data <new-dataset> --output <new-training>`.
Use `scripts/run_robotics.py --actor <new-training>/actor.pt --seeds 101 102 103
--output <new-evaluation>` for the fixed closed-loop suite. The suite exits after
saving `suite.json`; normal interactive mode stays available for reset.

The repository is a shared dirty worktree. No commit, push, deployment or hardware
operation was performed. This implementation's file hashes accompany the local
artifacts; unrelated edits were preserved.
