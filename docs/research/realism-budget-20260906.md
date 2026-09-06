# Realism and performance budget — 2026-09-06

The user wants realistic training observations and a convincing interactive view,
while preserving useful performance on the local Mac. This assessment measured
the existing renderer; it does not implement the proposed graphics or physics upgrades.

Implementation follow-up: [shared forest renderer and contacts](realism-implementation-20260906.md).
The measurements below describe the earlier baseline.

## Measured hardware and renderer cost

Live hardware inspection: Apple M4 Max, 14 CPU cores (10 performance, 4 efficiency),
32 GPU cores, 36 GiB unified memory. Metal 4 reported by macOS.

The actual forest builders, materials and sunrise lighting were measured through
Three.js 0.180.0 WebGPU in the Codex browser. Resolution was fixed at 1536 × 864,
pixel ratio 1, with the existing 4096² sun shadow map, antialiasing, 173,964
instanced canopy leaves and animated loose leaves. Each camera had 30 warmup
frames followed by 150 measured frames. GPU timestamp queries were required;
there was no replacement estimate when queries were unavailable.

| Camera | GPU median / p95 | CPU update + submission median / p95 | Serialized wall median / p95 |
| --- | --- | --- | --- |
| Drone departure | 4.33 / 4.98 ms | 4.60 / 5.40 ms | 14.50 / 24.40 ms |
| Cabin clearing | 3.67 / 4.85 ms | 4.80 / 5.50 ms | 16.90 / 23.30 ms |
| Forest overview | 4.19 / 6.55 ms | 4.70 / 5.30 ms | 17.00 / 24.20 ms |

GPU duration is the sum of timestamped shadow and color render passes. CPU timing
covers leaf updates and command submission. Serialized wall time additionally
includes timestamp readback, browser scheduling and the GPU completion wait.
It is deliberately not reported as achievable pipelined FPS. Draw/triangle counters
were reset before sampling and are omitted from the saved results.

At 30 display frames/s the total frame budget is 33.33 ms; at 60 it is 16.67 ms.
The single-view GPU cost leaves room for more visual work. It does not establish
a whole-system multiplier: three sensor cameras, depth/image readback, actor
inference, reconstruction and physical interactions were not in this probe.
The worst measured serialized p95 was already 24.4 ms.

The standalone forest displayed approximately 119 animation-loop frames/s before
the probe. That counter measures submission cadence. The integrated mission
viewer deliberately caps its render cycle at 30 Hz and waits for GPU completion.

Whole-machine readings during this investigation showed about 73–75% CPU idle,
and a 10-second GPU-counter sample ranged from 52–79% device utilization (median
69%). These include other applications and scene transitions, and cannot be
attributed to FlightRL. GPU queries above are the useful scene-specific evidence.
No swap was allocated or in use. macOS also showed substantial compressed memory;
36 GiB is shared by the CPU, GPU and other applications, not dedicated VRAM.

Evidence is under `artifacts/realism-budget-20260906/`: `forest-gpu.json`,
`gpu-system-samples.json`, `system-context.json`, `source.json`, and the temporary
HTML/TypeScript probe. To reproduce, copy the archived HTML to `viewer/` and the
TypeScript to `viewer/src/forest/`, open `/realism-profile.html` on the Vite server,
and click Run measured views after the module loads. Remove those two temporary
files afterwards. No production renderer or physics source changed in this pass.

## What exists and what is missing

- The detailed viewer already uses physically based material classes, procedural
  bark/soil textures, instancing, canopy shadows, fog and leaf animation. Geometry,
  texture variation and indirect illumination still look authored and repetitive.
- The native training camera traces simpler analytic geometry with procedural
  shading. Extra viewer branches, cabin and foliage do not automatically enter
  training RGB, sensor depth or collision tests. The current detailed exports are
  RGB re-renders, not images consumed by the trained actor.
- The drone has native six-degree-of-freedom dynamics and swept collision
  detection. Current missions terminate on collision; they do not solve general
  contact impulses, friction, bouncing and stacks of movable bodies.
- Dust already has persistent world positions, inertial drag, gravity/settling,
  swept contact, deposition and resuspension in a shared reduced-order airflow
  field. This is not a full fluid/pressure solver. Falling leaves in the detailed
  viewer are currently an animation, without general object-contact dynamics.

## Recommended implementation order

1. Establish one authoritative scene description for visible geometry, sensor RGB
   and depth, and collision proxies. Share object transforms, units and timestamps.
   Validate corresponding surface hits before claiming improved training fidelity.
   Small visual detail may use a simpler collision proxy, but the approximation
   must be deliberate; navigable obstacles cannot exist only in the viewer.
2. Upgrade one bounded forest area with scanned bark/ground textures and meshes,
   natural shape variation, proper normal/roughness maps, environment lighting,
   better foliage transmission and restrained contact shading. Use instancing,
   mipmaps, compressed textures, distance-dependent geometry and limited shadow
   work. Keep sensor exposure and motion effects explicit and tied to its timing.
3. Integrate an existing native 3D contact solver for the drone and nearby movable
   objects. Jolt is a suitable C++ candidate to evaluate against the native code;
   retain the measured drone forces and inertia. Validate thin-wall impacts,
   glancing contacts, friction, resting contact and fixed-step repeatability.
   Record the actual contact state for viewer replay instead of simulating a
   different physical world independently in JavaScript.
4. Simulate rain and airborne leaves as many inexpensive particles with gravity,
   drag and scene contact. Reserve full rigid bodies for objects whose rotation
   and contact forces matter. Couple dust emission to contact/flow events, with
   finite mass accounting where claimed. Only move a measured particle bottleneck
   to Metal/WebGPU compute; keep physics and sensor results synchronized.

An initial acceptance target is sustained 30 Hz presentation plus all three
RGB-D cameras at the existing 10 Hz sensor schedule, retaining the current physics
timestep. Measure completed image delivery, p95 latency and memory growth with the
policy and reconstruction running. Sixty Hz presentation is a stretch target
until that combined workload is measured. These are proposed gates, not achieved
new-renderer results.

## Reuse and engine boundary

Many components can be reused. Poly Haven offers CC0 scanned materials, models and
HDR environments. Three.js has WebGPU ambient-occlusion examples. Jolt and Rapier
provide established 3D rigid-body/contact capabilities. Box2D is for 2D dynamics.
On Apple silicon, Metal/WebGPU is the relevant GPU route; CUDA GPU physics targets
NVIDIA hardware and is not a direct acceleration path for this Mac.

An Unreal-like appearance in a bounded scene is a reasonable aim. Porting Unreal's
complete lighting, geometry streaming, temporal rendering and physics stack is
a separate engine-development project. Unreal source access is governed by Epic's
EULA; it is not permissively licensed source that can simply be copied here.
If the eventual requirement is Unreal's complete feature set, using Unreal as the
rendering backend is more credible than recreating it. Epic's current macOS
feature matrix also lists limits, including no hardware-ray-traced Lumen/MegaLights
and beta Nanite/virtual shadow support on M2+.

Primary references checked on 2026-09-06:

- [Apple Metal](https://developer.apple.com/metal/)
- [Three.js WebGPU ambient occlusion](https://threejs.org/examples/webgpu_postprocessing_ao.html)
- [Poly Haven license](https://polyhaven.com/license)
- [Jolt Physics](https://github.com/jrouwe/JoltPhysics)
- [Rapier rigid bodies](https://rapier.rs/docs/user_guides/javascript/rigid_bodies/)
- [PhysX GPU requirements](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/GPURigidBodies.html)
- [Epic source access](https://www.unrealengine.com/ue-on-github?lang=en-US)
- [Unreal macOS requirements](https://dev.epicgames.com/documentation/en-us/unreal-engine/macos-development-requirements-for-unreal-engine)
