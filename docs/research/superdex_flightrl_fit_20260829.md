# SuperDex fit for FlightRL

Assessment date: 2026-08-29. Official sources only. The latest published
`stable` branch inspected was commit
[`1d71509`](https://github.com/facebookresearch/project_superdex/tree/1d7150946fa3f3d3fb09c2bff07eaa138cbfdee6).
This is an architecture assessment, not a local reproduction, migration plan,
or hardware-authority record.

## Verdict

SuperDex is potentially useful to FlightRL as a **narrow experimental
contact/reference backend**, especially for landing, perching, entanglement,
or interaction with deformable objects. It should not replace the native
C/Ocean training spine or the current MuJoCo validation lane yet.

For ordinary free-flight navigation, SuperDex's distinctive strengths—dense
compliant contact, non-convex collision, and coupled rigid/deformable
simulation—do not address FlightRL's current limiting path. They do not provide
the missing edge-v3 training/lowering pipeline, real-camera transfer evidence,
GAP8 implementation, CPX protocol, STM32 safety runtime, or typed deployment
bundle described in [the current architecture](../architecture.md) and
[sim-to-real roadmap](../sim_to_real_roadmap.md).

The best next action is therefore a sealed benchmark spike only after a
contact-rich aerial task becomes important: implement one rigid quadrotor,
replay the same deterministic force/torque sequence in SuperDex and MuJoCo,
and compare state/contact error and wall-clock throughput. Do not port PPO,
camera training, or the live-transfer stack before that gate wins.

## What SuperDex actually offers

| Area | Verified official surface | FlightRL implication |
| --- | --- | --- |
| Physics | Rigid, soft, and articulated actors are stable; shell and rod actors are experimental. Contact is compliant, uses signed-distance/generalized collider fields, and produces surface traction distributions rather than only point-resultant forces. Some penetration is inherent; direct triangle-mesh contact and deforming soft colliders carry experimental/performance caveats. [Physics overview](https://projectsuperdex.com/physics/docs/overview/), [actor status](https://projectsuperdex.com/physics/docs/concepts/actors/overview/), [contact model](https://projectsuperdex.com/physics/docs/concepts/contact/) | Materially stronger reference lane for contact-rich or deformable interaction. This is mostly excess complexity for open-space Crazyflie flight. The X post's broad MuJoCo contrast is not itself evidence of a FlightRL advantage. |
| Aerial rigid body | A dynamic rigid actor has six DoF and accepts explicit mass, center of mass, inertia, linear/angular velocity, and world-frame generalized force/torque. Arbitrary triangle-mesh geometry is supported. The default finite-step rotation scheme may add noticeable dissipation and does not exactly conserve angular momentum/energy; an alternative Newton-Euler scheme is experimental. [Rigid actors](https://projectsuperdex.com/physics/docs/concepts/actors/rigid_actors/), [force API](https://projectsuperdex.com/physics/generated/api/v1.0/python/api/physics.html) | A custom quadrotor is feasible, but not supplied. FlightRL would still need rotor thrust/torques, motor lag, drag, action frames, sensors, resets, rewards, and timestep-level trajectory parity tests. |
| Robotics extensibility | Robots can be defined/imported from URDF; controller, sensor, and actuator types are registerable. The initial built-ins are joint-space/operational-space controllers and a camera descriptor. No concrete actuator type ships, and applications wire sensing/control/application order themselves. [Robotics overview](https://projectsuperdex.com/robotics/docs/overview/), [components](https://projectsuperdex.com/robotics/docs/bot_components/) | The framework does not block a drone, but it does not provide a rotorcraft or flight-control stack. Integration cost is substantial and would duplicate contracts already implemented in FlightRL. |
| RL/API | SuperDex Lab exposes Gymnasium-compatible custom environments and Ray/RLlib PPO/SAC examples. Custom environments define action application, observation, rewards, stop criteria, and simulation/control rates. The initial catalog lists CartPole, Ant, and HalfCheetah, not an aerial task. [Lab introduction](https://projectsuperdex.com/lab/docs/superdex_gym/intro/), [custom environments](https://projectsuperdex.com/lab/docs/superdex_gym/environments/) | Usable for an independent reference experiment, but it is not an Ocean/Puffer export and provides no checkpoint compatibility with FlightRL. |
| Vectorization | `HybridVectorEnv` uses asynchronous subprocesses outside and sequential `SyncVectorEnv` stepping inside each worker, with shared scenes. Official benchmarks force the physics solver to one thread and obtain parallelism from the vectorization layers. [Large batches](https://projectsuperdex.com/lab/docs/superdex_gym/batching/), [benchmark contract](https://projectsuperdex.com/lab/docs/superdex_gym/benchmarking/) | “Scalable” should not be read as proof that it beats FlightRL's contiguous native C stepping. Throughput and memory must be measured on the exact drone scene before any training-path decision; the inspected official material did not establish GPU-physics acceleration. |
| Sensors/rendering | The component framework can host custom sensors. The shipped `SENSOR_CAMERA` stores pose/intrinsics but produces nothing in the physics loop; a renderer consumes its parameters. Lab rendering uses Polyscope for `human` or `rgb_array` output and supports only one rendered environment per process. [Components](https://projectsuperdex.com/robotics/docs/bot_components/), [Lab rendering](https://projectsuperdex.com/lab/docs/superdex_gym/render/) | Useful for geometry/contact debugging and videos, not evidence that SuperDex closes FlightRL's renderer-to-AI-Deck appearance gap or supplies IMU/ranger/camera noise calibrated to real hardware. |
| Platform | The supported release targets Linux x86-64, Windows x86-64, and macOS arm64 with C++ and Python APIs. Published wheels currently require Python 3.12. Linux GUI tools require X11/XWayland and OpenGL 4.1; macOS source builds require Xcode command-line tools. [Project README](https://github.com/facebookresearch/project_superdex#requirements) | It is viable for a Mac-side spike, but would add a separate Python 3.12/native dependency surface to FlightRL. Official support is not the same as a verified run in this repository. |
| Maturity/license | The changelog labels 2026-08-24 as the initial release, while the project README calls SuperDex Lab an early preview and says substantial improvements are planned. First-party code is Apache-2.0; docs/assets are generally CC-BY-4.0, the optional mesh CLI is GPLv3, and some third-party dependencies/assets have non-commercial or academic-only terms. [Changelog](https://github.com/facebookresearch/project_superdex/blob/stable/CHANGELOG.md), [README maturity and license](https://github.com/facebookresearch/project_superdex#superdex-lab-early-preview) | Treat APIs and performance as unproven for FlightRL. First-party code is permissive, but every imported asset/dependency needs a component-level license check before commercial reuse. |

## Fit against FlightRL's active lanes

### Native C/Ocean training spine: poor replacement fit

FlightRL's native environment is deliberately dependency-light, vectorized in
place, and already carries the 28-value observation, four-action control,
domain-randomization, task, and evidence contracts described in
[the backend guide](../backend_usage.md) and
[Crazyflie environment note](../pufferlib_crazyflie_env.md). SuperDex offers a
friendlier general physics/authoring surface, but its documented vectorization
is process-based and sequential within workers. There is no current evidence
that migrating would improve the dominant training workload.

### MuJoCo validation lane: complementary only for new contact questions

MuJoCo already covers FlightRL's rigid-body dynamics, forbidden contacts, room
geometry, range semantics, and AI Deck rendering checks. SuperDex becomes
meaningfully complementary when the question depends on non-convex distributed
contact or deformation—for example a compliant landing surface, perching pad,
cable/net interaction, cloth, or soft obstacle. It does not justify replacing
MuJoCo for the current rigid indoor-navigation contract.

### Camera and edge deployment: no blocker relief

SuperDex's debug/rgb rendering and camera descriptor do not supply a calibrated
AI Deck visual sensor model. More importantly, a desktop simulator cannot close
the current float-C/int8/GAP8, CPX, STM32 safety, evidence-binding, and hardware
promotion gaps. Those remain separate systems work even if a SuperDex policy
trains successfully.

## Promotion gate for any future spike

Proceed beyond a reference experiment only if all of these are true:

1. A named FlightRL task genuinely requires distributed/non-convex or
   deformable contact; generic navigation is insufficient justification.
2. A minimal rigid quadrotor reproduces the same no-contact trajectory as the
   current reference within declared pose/velocity tolerances.
3. The contact scenario produces useful independent evidence that MuJoCo and
   the native approximation cannot provide as cheaply.
4. Measured step throughput and memory on the Mac are acceptable for the
   intended evaluation batch.
5. Observation/action units, frames, resets, termination, and artifact
   provenance remain explicit; no SuperDex checkpoint is relabeled as an
   edge-v3 or hardware-approved artifact.

Until then, keep SuperDex on the watchlist as a promising, very new
contact-physics platform—not a current FlightRL blocker fix.
