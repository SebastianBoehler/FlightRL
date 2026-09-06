# Industrial site expansion — 2026-09-06

Continues the [first mixed-robot milestone](robotics-inspection-20260906.md).
This implements power-utility and production-utility site families, noisy sensing,
a learned visual-servo controller for both robots, camera-derived inspection
handovers, and repeatable closed-loop evaluation. This is a simulator prototype,
not a calibrated facility digital twin or a deployment-ready autonomy system.

## China-trip scenario

The September 1 BW-i preparation notes describe a five-minute China-market pitch,
Shanghai ecosystem/manufacturing-network meetings, Taicang industrial firms and
Suzhou companies/startups. They support demonstrating industrial inspection and
seeking hardware/manufacturing partners. They do not confirm a mine or power-station
visit. The authored sites are representative customer scenarios, not named hosts.

Power utilities and production plants are credible inspection settings: equipment
status, gauges, elevated assets and repeatable inspection rounds appear in current
industrial offerings. See [ANYbotics power utilities](https://www.anybotics.com/industries/robotic-inspections-for-power-utilities/),
[DJI substation inspection](https://enterprise.dji.com/inspection/substation-inspection?from=nav&site=enterprise)
and [Siemens inspection automation](https://www.siemens.com/en-us/products/anybotics-inspection-automation/).
Our implemented task reads red/green status indicators beside fiducials. It does
not infer electrical faults, temperature, gas concentration or equipment health.

A five-minute trip demonstration can show the site overview, run elevated
inspection and ground verification with both live cameras, interrupt/restore the
robot link, then show docking evidence and switch to the production layout.
Seed 162 provides a red status indicator in the power site. The partner ask is
concrete: one recurring inspection task, access to the asset and sensor recordings,
and a reference robot whose mechanics can be identified. This keeps the discussion
about operational requirements and measurable validation.

## What changed

- Seeded power and production variants: transformers with radiator fins and
  bushings, pumps, pipe racks, conveyors/press cells, gantry, tanks, stacks,
  cladding and service lanes. Equipment spacing, target positions/IDs, initial
  poses, friction, wind and lighting vary within an authored site grammar.
- Rendering instances repeated geometry and shares compiled MuJoCo collision
  shapes. Materials, concrete texture, sunlight and atmospheric haze provide the
  site appearance. The observer retains free movement, direction guides/trails
  and the two live camera views.
- The drone observes an elevated asset's marker and signal, then sends its
  measured status, image hash, estimated location and nominal position variance.
  The rover waits for delivery, consumes the requested follow-up marker, verifies
  the lower equipment signal and approaches the docking marker.
- Marker location and surface normal come from actual rendered RGB-D. The
  controller never receives authored target coordinates. Those coordinates are
  available to scenario generation and the separate report evaluator.
- Depth noise grows with range, 1% of pixels are missing, camera delivery is one
  batch late, and two scheduled outages remove observations. Proprioception has
  seeded bias/noise; rover translation is integrated from wheel encoders.
  Drone velocity is a modeled noisy velocity estimate, not implemented VIO.
- A physical maintenance screen temporarily occludes the route. Its scheduled
  position change is a test fixture, not calibrated gate dynamics. Clearance
  braking and missing-observation handling are explicit shared control logic.
- Inspection is independently checked against camera-mount position and signal
  truth. Equipment contact disqualifies and stops a mission. After inspection,
  the drone continues visual station-keeping instead of holding zero velocity
  indefinitely on biased estimates.

The handover's location is an odometric estimate from a known launch reference;
its uncertainty is a nominal model, not empirically calibrated covariance. The
rover uses the delivered marker ID for local visual control. General map fusion,
route planning from uncertain coordinates and autonomous work-order generation
are not implemented.

## Learning and environment contract

`RobotEnvironment` exposes reset, physical stepping, camera observation delivery,
reward and episode outcomes independently of the browser transport. The browser
remains the actual RGB-D provider. `--fast` advances virtual time after each
rendered sensor batch: 500 Hz physics, 50 Hz control, 10 Hz virtual sensing.
It is accelerated serial evaluation, not headless batched GPU image training.

The new 5,828-parameter shared MLP receives 21 features: detected relative marker
position/normal, visibility, three depth clearances, nine proprioceptive values
and robot role. It emits velocity/yaw setpoints. AprilTag detection, indicator
reading, task sequencing, clearance limits and report criteria remain explicit
code. This is imitation learning of visual servo commands for both embodiments,
not end-to-end pixel RL or a general learned planner. The earlier raw RGB-D drone
actor remains a separate contract for the original corridor.

Splits were fixed before fitting: train seeds 120–125 (3,000 robot examples),
validation 140–141 (600), closed-loop test 160–167. Training poses vary distance,
lateral/vertical offset, orientation and velocity. They are rendered static
observations, not expert full-episode trajectories. The site families share an
inspection-lane structure; held-out seeds do not establish generalization to
arbitrary buildings or robot mechanics.

Apple MPS training took 4.74 seconds, selected epoch 346 of 350 using validation
alone. Validation command MAE: 0.0049/0.0045/0.0028 m/s XYZ and 0.0079 rad/s yaw.
Frozen actor SHA-256:
`c9ea357c022ed64555646dd08686c25dc3061c6627cbfb963948cdc3bbc06309`.

## Evaluation and retained failures

| Final check | Result |
| --- | --- |
| Classical controller, test seeds 160–167 | 8/8 missions, 24/24 verified reports, zero equipment-contact steps |
| Frozen learned pair, same seeds/settings | 8/8 missions, 24/24 verified reports, zero equipment-contact steps |
| Mean simulated mission duration | 89.21 s classical; 89.20 s learned |
| Largest final inspection/docking position error | 0.145 m, below the declared 0.25 m drone / 0.22 m rover scoring limits |
| Scheduled camera outages | Seven missing batches per mission, recovered in all 16 final tests |
| Serial visual evaluation | Median 2.71× real time classical, 2.88× learned |
| Raw render/physics depth audit | 4,607/4,608 rays within 6 mm; one ground sample differed by 15.65 mm |
| Focused tests and build | 18 Python tests, Ruff and TypeScript/Vite build passed |

The actor matched this classical baseline's success and mean duration; these
results do not establish superiority. Five test seeds have red status indicators
and three green. The eight variations cover two authored site families, not eight
independently designed facilities. Source reports retain every event, image hash,
scene description, controller identity, sensor errors and contact count.

Depth audit samples both robots at three capture times per scenario, using the
actual raw camera buffers before modeled noise. The final classical maximum was
1.31 mm; one learned ground sample was 15.65 mm. That exception and its neighboring
pixels are retained, so this is not a universal sub-6-mm parity claim. Increasing
cylinder tessellation from 20 to 96 sides removed the earlier large silhouette
errors, including a 1.31 m missed thin-insulator sample. The actor was not retrained
or selected using these tests; the final suites reran after the rendering change.

An earlier `sensors-baseline-v2` run had three correct readings but 35,489 physics
steps with equipment contact: the drone drifted after inspection while holding
zero velocity on biased estimates. That run is invalidated by the final contact
gate. Visual station-keeping remains active after inspection, and a regression
checks it. Failed/partial development artifacts are retained rather than counted
as successful demonstrations.

On this Apple M4 Max (14 CPU cores, 36 GB), the isolated serial CPU benchmark ran
about 1,164–1,191 control steps/s, or 11,635–11,912 physical substeps/s, across batches
of 1/4/8 environments. That is 23.3–23.8× aggregate simulated real time. Reset took
17–44 ms per world; peak benchmark-process RSS was 107 MiB. These measurements
exclude camera rendering, the browser and visual-policy execution. They do not
represent whole-Mac memory or GPU utilization. The visual path is the tighter
throughput limit, and the observer remains capped at 30 fps.

The normal-speed, 1536×864 link-recovery run (`interactive-final/1df76852`)
completed all three checks in 115.42 simulated seconds with zero equipment contact.
The link was unavailable from 0.28 s to 57.58 s; the rover visibly remained in
`await_task` until delivery. Median display rate was 30.00 fps and camera cadence
9.99 batches/s. End-to-end camera latency p95 was 46.67 ms against a nominal
100 ms period; physics/control p95 was 1.94 ms per 20 ms update. This establishes
measured scheduling headroom for this configuration, not a GPU-utilization
percentage or linear robot-count scaling promise. The prior 48.8 s interactive
capture was reset before completion and is not counted as a mission success.
Both site-selector directions and the overview/equipment camera buttons were
checked in the live UI; production-view screenshots are saved with the artifacts.
Site-switch previews are partial episodes, not extra evaluation successes.

Artifacts: `final-baseline/suite.json`, `final-learned/suite.json`,
`cpu-throughput.json`, `assessment.json`, `tests.log` and `build.log` under the
artifact root. `source-manifest.json` binds the implementation, shared sensor code
and marker images to their SHA-256 hashes. The existing Three.js chunk-size build warning remains.

## Reproduce

From this checkout, use its existing `.venv` and viewer dependencies:

```sh
npm run dev --prefix viewer -- --port 4173
PYTHONPATH=src .venv/bin/python scripts/run_robotics.py --industry \
  --actor artifacts/industry-expansion-20260906/training/actor.pt \
  --output artifacts/industry-live
```

Open `http://127.0.0.1:4173/robotics.html`. Omit `--actor` for the classical
baseline. The Industrial site selector switches between power (seed 0) and
production (seed 1), starting a fresh episode. `--seed` selects other variations
when the URL has no site selection. Use a new output root per experiment; each connection saves a separate
episode. The original corridor remains available without `--industry`.

Stop the bridge before collection. Run `scripts/collect_industry.py --output
<new-dataset>`, open the robotics page once ready, and keep it visible until both
splits are saved. Run `scripts/train_industry.py --data <new-dataset> --output
<new-training>`. All Python commands use `PYTHONPATH=src .venv/bin/python`.

For frozen evaluation, add `--fast --seeds 160 161 162 163 164 165 166 167` to the
bridge command and use a new output root. It saves individual reports and a
`suite.json`, then exits. Do not use those test seeds to select a checkpoint.

## Scope and next step

Shared robotics materials and primitive tessellation changed in this expansion.
The earlier corridor's recorded results remain historical evidence, not a new
validation of that raw-pixel checkpoint under the changed renderer.

Models use surrogate mechanical parameters and nominal sensor errors. This does
not add a validated IMU/VIO estimator, suspension/tire model, mine terrain,
particulate sensor degradation, articulated robots or marine mechanics. The next
useful extension is a specific partner's robot and inspection asset, with measured
sensor/dynamics data and a task failure envelope. Mining should become a separate
terrain/dust scenario when the actual customer task warrants it.

Local artifacts live in `artifacts/industry-expansion-20260906/`; large datasets,
weights and recordings are gitignored. No hardware, deployment or paid compute
was used. Unrelated worktree changes were preserved.
