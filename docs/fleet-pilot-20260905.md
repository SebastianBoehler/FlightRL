# Three-drone transfer pilot — 2026-09-05

Results: {'collision': 21, 'budget_exhausted': 3, 'incompatible_envelope': 30} across 54 held-out combinations. Zero complete missions.
The experiment is rejected as a usable cooperative controller.

## Implemented contract

Three drones advance together through native six-DOF integration. Each gets a local
64×48 RGB-D image including peer collision envelopes. Peer messages contain estimated
pose, measured velocity, static target assignment, completion IDs and timestamp.
Messages arrive after 200 ms, publish at 5 Hz, expire after 1 s, and exclude self.
The learned shared controller consumes its own image, vehicle/motion/goal features,
the nearest two peer messages and a mission identifier. A camera brake remains explicit.
No raw peer images or simulator truth corrections enter the local policy. Odometry is
integrated noisy velocity with known spawn origin; this is not VIO.

Known goal coordinates and fixed per-drone assignments are supplied. The missions are
waypoint dwell (inspection surrogate, not visual defect inspection), arrival (delivery
surrogate, no payload), and return to spawn after an 8-second communications cutoff.
Completion messages are transported but do not yet trigger task reassignment.

## Training and evaluation

Behavior cloning on utility-plant/data-center layouts, seed 20, FPV/industrial references,
waypoint dwell and arrival. The poor teacher trajectories were retained as exploratory
data; their collisions explain why this checkpoint must not be promoted.
12 epochs with fixed seed; checkpoint saved before tests. Test seeds 120 and 121 include
forest, agriculture vehicle and return mission holdouts. Unsupported size combinations
are reported explicitly, not counted as successful flight. The first attempt failed on
an unsupported MPS pooling operation; v2 uses fixed pooling and completed.

Compatible rollout throughput: 898.0–1268.6 agent camera steps/s
(mean 1064.9); three agent observations per world tick. This includes
camera rendering, controller inference and native integration, but not detailed WebGPU
rendering, particles or inter-drone wake interactions. It is not comparable to the full
dust benchmark without accounting for those workloads.

## Vehicle references and limits

Avata 2: published 0.377 kg, 0.185×0.212×0.064 m.
Matrice 350: published 6.47 kg with batteries; assumed conservative propeller envelope.
Agras T25: published 32 kg with battery and no payload, 2.585×2.675×0.780 m unfolded.
Sources are included per vehicle in the frozen plan. Motor and rate response constants
are explicit assumptions, not manufacturer measurements. Collision uses a conservative
orientation-independent sphere. This is not a calibrated DJI digital twin or validated
cross-airframe transfer. Agricultural flight requires a larger outdoor scenario.

## Outstanding integration

The detailed forest now renders in both mission camera panels, at recorded poses.
This fleet training still uses native analytic geometry. Detailed mesh/depth training,
verified vehicle dynamics, rotor-area-derived wakes, peer aerodynamic coupling and
robust multi-mission cooperation remain outstanding. Improve the teacher on training
layouts and add known-map routing/clearance tests before more imitation training.

Artifacts: artifacts/fleet-pilot-20260905-v2 (plan, checkpoint, source hashes, results,
and actual three-drone forest replay). 23 focused environment/link tests and viewer build pass.

## Fleet dashboard

`/fleet.html`, linked from the main replay header, displays the actual saved
98-frame, three-FPV-drone forest run. It includes color-coded aircraft and trails,
assigned targets, selection/follow controls, three simultaneous camera viewports,
scrubbing, optional looping, and the collision outcome. A single WebGPU renderer
submits all four views for the same recorded state before yielding.

The original images and time-resolved peer-link state were not saved in this
artifact. Cameras are labeled detailed visual re-renders; the communication label
states the experiment contract, not live per-drone connectivity. The dashboard
does not imply a successful swarm policy or new training. Future runs still need
better teachers, validated clearance and explicit per-drone failure records.

Reproduce the public bundle with `PYTHONPATH=src .venv/bin/python
scripts/package_fleet_review.py`; this verifies both spawn positions and goals
against the reconstructed forest seed 120, and stores the source replay SHA-256.
