# Robotics platform direction — 2026-09-06

Status: proposal after reviewing the current implementation, the September 2
cross-airframe architecture and September 3 technology trajectory. That review did not implement robot
adapters or training runs. The subsequently approved first milestone is now
implemented and documented in [mixed-robot inspection](../robotics-inspection-20260906.md).
The next expansion adds [varied industrial sites and learned drone/rover control](../industry-expansion-20260906.md).

## Product direction

Build a robotics training and validation workbench: describe a robot, its sensors,
a world and a task; produce reproducible training episodes, policies and evidence.
FlightRL supplies the first flight embodiment. The shared asset is the experiment
and validation pipeline; each new robot still needs identified mechanics,
actuators, observation/action contracts and measured operating limits.

A robot is a physical body (possibly many articulated links), actuators, sensors,
onboard computation/communications and energy constraints. “Flying sensors” is a
useful perception intuition, but actions change both motion and future observations.
Motor limits, slip, contact, latency and observability can dominate transfer even
when the rendered image is convincing.

The strongest initial product example is mixed-robot industrial inspection:
a drone inspects elevated equipment; a ground robot inspects under pipes, reads
instruments and docks. They exchange observations and task status under degraded
communications. This connects the existing industrial mission work with the new
shared rendering/contact scene, instead of building unrelated robot showpieces.
The first customer hypothesis is inspection/autonomy teams comparing sensor rigs,
controllers and failure recovery; willingness to pay has not been established.

## Current code versus the earlier architecture

- `scenario_bundle.py` already binds scenes and provenance, but its compiler still
  requires SixDofPhysicsProfile, SixDofSensorProfile and BoxRoom. It is not a
  generic articulated robot or arbitrary-terrain compiler.
- `policy_io_contract.py` provides useful typed signal contracts; its action modes
  remain velocity/yaw, body-rate/thrust and direct motor thrust. It needs distinct
  wheel/steering, joint target/torque and thruster contracts. Changing enum names
  alone would not make the actor/trainer/runtime support these interfaces.
- `src/flightrl/realism/` and `viewer/src/realism/` provide a real shared world,
  rendered RGB-D, native contacts, motion history and saved sensor observations.
  They hardcode three drones, two image resolutions and drone actuation. The
  Python/WebSocket execution path is a useful interactive validation loop, not
  the existing highly batched native training loop.
- `src/flightrl/mujoco/` already provides an independent flight/reference lane.
  There is no demonstrated general quadruped, humanoid, rover or marine robot here.
- The current dust and position-hold controllers use simulator state explicitly.
  The frozen RGB-D actor has not been retrained for this forest. A successful
  scripted demonstration cannot stand in for held-out learned-policy success.

Two earlier recommendations now need deliberate revision. “MuJoCo is reference
only” was a flight-specific throughput decision; it should not force us to invent
an articulated-body solver. “WebGPU is display only” no longer describes the live
RGB-D implementation. Preserve the distinction between an interactive sensor path
and a validated batch training path, rather than pretending the first scales to
thousands of environments.

## Shared capability map

| Capability | Next concrete addition | Acceptance evidence |
| --- | --- | --- |
| Robot description | RobotSpec: link/joint graph, inertias, frames, geometry, actuator limits, sensor mounts; one explicit importer initially | Imported mass, joint axes, transforms and inertias checked against source; unsupported fields rejected |
| Physics ownership | Backend selected per interacting world; authoritative state and contacts exposed through a common interface | No body integrated twice; force/energy/contact tests and timestep convergence |
| Sensors and clocks | Calibrated RGB-D, IMU, wheel/joint encoders; then LiDAR and contact/force sensing; latency, validity and mounting transforms | Golden geometry/time tests and representative real sensor recordings |
| Markers and perception | Visual fiducials for docking/inspection, observed through rendered images with occlusion and lighting variation | Detector errors and pose errors measured; hidden marker IDs/poses never fed to actor |
| Actions and controllers | Separate setpoints, wheel speeds/steering, joint position/torque, thrusters; explicit lower-level controller and actuator lag | Action units, limits, update rate and delay validated per robot |
| Tasks and cooperation | Search, inspect, approach, dock, hand over, return; independent evaluator and delayed neighbor observations | Held-out tasks, false reports, communication loss and recovery measured |
| Training and evaluation | Reset/step/observe/reward/terminate contract; reproducible seeds and immutable splits; classical baseline first | Frozen policy succeeds on unseen layouts/dynamics; full rollout throughput and memory measured |
| Calibration and transfer | Fit dynamics/sensor uncertainty from logs; randomize measured ranges; compare prediction against hardware | Declared error envelope and separate simulator, software-in-loop and physical evidence |

URDF/MJCF/SDF are potential import formats, not three new runtimes to implement
at once. Choose the first format for the selected real robot and validate import
semantics. glTF remains useful for appearance; robot mechanics require explicit
inertias/joints/actuators beyond a visual mesh.

## Physics by robot family

| Family | Mechanics that must be represented | Suggested order |
| --- | --- | --- |
| Drone | Calibrated thrust/torque, actuator lag, inertia, drag/wind, battery effects, landing/contact | Close current sensor-policy loop first |
| Wheeled rover / car | Wheel joints, motor/brake torque, traction and slip, steering geometry; suspension where task needs it | First new embodiment; low-speed rover before high-speed tire dynamics |
| Arm | Joint limits, drive dynamics, gripper/contact friction, payload inertia and force sensing | Small manipulation task can test articulation before legs |
| Quadruped | Articulated inertia, foot contact/slip, torque-speed limits, proprioception and terrain compliance | Next mobility extension if uneven terrain is the product focus |
| Humanoid | Whole-body contact/balance, self-collision, hands/manipulation, actuator/thermal limits | After articulation and contact validation |
| Surface vessel | Buoyancy, water-relative drag, thrusters/rudder, waves/current and docking contact | Separate family when a waterfront task justifies it |
| Underwater vehicle | Added mass, hydrodynamic damping, thruster response, pressure/sonar and underwater optical effects | Beyond the surface-vessel step |
| High-speed aerial vehicle | Identified aerodynamic envelope, high-rate control/actuator response, latency and reliable fast contacts | After ordinary flight control/sensing are validated |

Dust, rain and wind belong to world/sensor models, but should affect each robot
through the appropriate mechanism. Dust affects optics; wet terrain may affect
traction; current affects water-relative forces. Do not invent a universal
“bad weather force.” Full CFD, granular soil and every-grain collisions should
wait until a measured task failure needs that complexity.

## Engine and performance choices

Retain native flight dynamics and Jolt where already verified. For articulated
robots, evaluate the existing MuJoCo route first: it provides joint-coordinate
dynamics, actuators, constraints and frictional contact. This is a recommendation
to integrate and benchmark, not a claim that its contact model is universally
correct. [MuJoCo computation](https://mujoco.readthedocs.io/en/stable/computation/index.html)

A mixed drone/rover/arm world should have one solver own all bodies that physically
interact. For example, a MuJoCo world can receive our modeled rotor forces while
also solving a robot's joints and contacts. Independently integrating the drone
in Jolt and an interacting arm in MuJoCo would require a real co-simulation scheme;
matching their displayed positions does not solve contact exchange.

Use existing marine formulations as references if that family is selected.
Gazebo documents added mass, Coriolis terms, relative-current velocity and damping;
these go beyond adding buoyancy to a flying rigid body.
[Gazebo hydrodynamics](https://gazebosim.org/api/sim/9/theory_hydrodynamics.html)

Keep WebGPU/Metal for the present Mac rendering workload. Newton is worth a later
NVIDIA batch-training evaluation, but its current requirements explicitly make
macOS CPU-only, so it is not an immediate Apple GPU performance upgrade.
[Newton requirements](https://github.com/newton-physics/newton#requirements)

Keep three independent rates: display, sensor/policy scheduling and robot-specific
physics substeps. The measured 30 FPS / three cameras at 10 Hz result does not
establish legged dynamics throughput or the cost of hundreds of training worlds.
Benchmark complete environment-to-learner throughput, reset cost, p95 latency and
memory at increasing batch sizes. One-body C physics can remain an efficient
specialization; it need not become the semantic solver for every jointed robot.

Use a headless training execution path and the detailed live validation view
against the same compiled task/sensor definitions. Any lower-cost geometry or
photometry approximation must be explicit and tested; display quality is not a
universal accuracy setting.

## Recommended next milestones

1. Close one learned drone task in the current scene: fixed sensing, a baseline,
   frozen train/validation/test split, retrained actor, measured success and failures.
2. Add one low-speed wheeled rover to an industrial scene at the current visual
   standard. Shared RGB-D/IMU timing, wheel odometry, friction, docking and visual
   markers. AprilTag is a reusable detector, not an oracle for authored marker pose.
   [AprilTag implementation](https://github.com/AprilRobotics/apriltag)
3. Demonstrate drone/rover inspection with complementary viewpoints, map/task
   handover, intermittent links and independent verification. Report what each
   robot actually observed, not shared omniscient geometry.
4. Add either an arm or quadruped according to the chosen task. Keep locomotion
   and mission/perception learning separable initially; import a known robot model
   and train/evaluate its motor policy under its own action and timing contract.
5. Add marine or humanoid support only with a concrete task, reference robot and
   calibration/evaluation data. Build one measured family extension at a time.

The product's defensible value would be faster, reproducible decisions about
robots, sensors and policies across representative conditions. Render quality
makes those decisions inspectable; identified dynamics, causal sensing and
held-out results make them trustworthy. This proposal does not promise one neural
policy will transfer unchanged across flying, walking, driving and swimming.
