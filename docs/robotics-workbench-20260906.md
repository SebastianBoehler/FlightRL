# Imported arm and timestamped robotics workbench

2026-09-06. Local simulation implementation following the robotics retrospective.

## Run and inspect

From the repository checkout, with the existing native extension installed:

```sh
uv pip install --python .venv/bin/python '.[robotics]'
PYTHONPATH=src .venv/bin/python scripts/run_robotics.py \
  --output artifacts/robotics-workbench-local --industry --arm \
  --actor artifacts/industry-expansion-20260906/training/actor.pt
```

The output directory must be new. The optional actor flag selects the existing
learned drone/rover controller; omit it for the visual baseline. Start the viewer
with its existing Vite command and open `/robotics.html?site=production`.

1. Select xArm7 and Focus selected robot. Open its joint controls, enter a valid
   arm target in radians, and Apply arm setpoints. The gripper uses the source
   model's control scale. Invalid values are rejected with a visible error.
2. Pause & save finalizes the episode. Drag Recorded capture to see the stored
   RGB images, physical body poses, joint readings and control targets from that
   acquisition. The plot cursor follows the selected capture.
3. Latest capture seeks to the last recorded frame; it does not resume physics.
   Reset mission starts a new episode. Switching sites also starts a new episode.

## Imported mechanics

The pinned MuJoCo Menagerie xArm7 is attached through MjSpec. It preserves seven
arm joints, six gripper joints, eight actuators, the fixed tendon, equality
constraints, inertias, joint/control/force limits and original gain/bias settings.
The scene renders its original compiled visual meshes. Collision proxies remain
in physics. A wrist camera joins the drone and rover cameras.

`assets/robots/xarm7/manifest.json` records every source file hash, upstream commit
and BSD-3-Clause licence. `scripts/fetch_xarm7.py` reproduces the asset fetch and
checks upstream blob hashes; it requires a nonexistent destination directory.
The source identity is checked before import and included in the episode.

This is a real reference-model integration, not an arbitrary URDF upload UI or a
calibrated agricultural airframe. The arm runs reference joint servos with manual
setpoints. No arm policy, manipulation task or hardware controller was trained.

## Time and recording contract

All simulation streams use integer ticks and episode-relative nanoseconds.
Acquisition time, sensor availability, decision time and control application time
are distinct. Inspection truth scoring uses the captured camera pose. Training
samples now pair commands with the delayed observation that caused them.
The modeled delay selects the newest capture at least 100 ms old, records its
actual age and counts skipped observations; it rejects acquisition-clock reversal.

`run.mcap` records 50 Hz physical state and actuation, every raw RGB-D capture at
both resolutions, delivered noisy observations, decisions and model/scene identity.
Actuation links to its source observation and records application-time feedback.
MCAP log/publish times are simulation availability times. Payload acquisition
fields retain sensor time. Host monotonic receipt time is labeled separately.

The UI shows raw acquisitions. Delayed controller observations and decision
records are retained separately; they are not presented as equivalent images.
Replay loads original RGB pixels and stored transforms instead of rerendering
historical evidence. It performs an indexed seek and rejects missing streams.
The custom RGB-D payload is an NPZ schema, not a standard ROS image channel;
MCAP compatibility does not imply automatic Foxglove panel support.

The recorder uses a bounded asynchronous queue and stops explicitly on recording
failure instead of silently dropping evidence. Full-resolution float depth is
expensive to retain: the complete measured episode used 5.14 GB in 128.7 seconds.
Long recordings need a measured retention/encoding strategy before deployment.

## Verification and measured limits

- 24 focused tests pass: existing robotics/industry behavior, captured-pose
  scoring, jittered delivery, causal training/actuation IDs, exact RGB/depth
  round trips, source model constraints, camera transforms and recorder failure.
- TypeScript and production viewer build pass. Existing large Three.js chunk
  warnings remain. Scoped Python error/undefined-name checks pass.
- One complete learned production episode with the arm attached passed all three
  inspections/docking checks at 128.74 simulation seconds, with zero counted
  drone/rover equipment-collision steps. Manual joint 1 target 0.4 rad settled at
  0.400 rad. This is one episode, not a new multi-seed qualification.
- Retained report: `artifacts/robotics-workbench-20260906/live-v3/c46d0365/report.json`.
  Display median 30.00 FPS; camera batch p95 65.91 ms against a 100 ms period;
  physics/control p95 3.68 ms against a 20 ms period. 1,282 capture batches and
  30,801 MCAP messages; recorder queue peak 16 of 64.
- Three sampled depth audits each compared 144 rays. Every sampled ray was
  within 6 mm of the physical raycast. This does not certify all mesh contacts.
- Final control/replay check: `final/8d4ce7d1` under the same artifact root,
  paused deliberately at 71.52 seconds. Invalid setpoint rejected, valid command
  recovered, first/last capture replay restored 0/0.4 rad targets and positions.
- Browser console error log empty. Narrow-layout check at 390 px requested
  viewport had no horizontal overflow; temporary viewport override was reset.

These timings are component latencies, not a measurement of total Mac GPU/CPU
utilization or a guarantee of additive headroom. No thermal soak, hardware clock
synchronization, ROS interoperability certification or hardware transfer was run.
Old pre-fix success reports retain their historical timing limitations.

## Product direction

See [positioning and partner milestone](research/robotics-product-positioning-20260906.md).
The next bounded implementation should be an arm task with independent success
and contact criteria, followed by a reference baseline and policy evaluation.
