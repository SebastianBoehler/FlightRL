# FlightRL cross-airframe autonomy co-design stack

**Status:** architectural recommendation, 2026-09-02

**Repository snapshot inspected:** `1ac9a0c1d63ab6e3781bf5cfd2c8873521d462fc` plus an unrelated dirty worktree

**Decision horizon:** local development now; Apple Silicon and x86 CPU first, NVIDIA scale-out later, embedded C/INT8 and FPGA as export targets

## Executive decision

Keep FlightRL's existing split and make it explicit:

- **C17 is the semantic authority** for deterministic batched dynamics, observations, rewards, resets, safety logic, and the deployable actor runtime.
- **C++20 is an implementation language behind a C ABI**, used only where a vendor ecosystem requires it: CUDA, Vitis HLS, and complex host-side accelerator code. Use Objective-C++ only as the thin host bridge to Metal.
- **Metal Shading Language is the first GPU backend**, initially for batched low-resolution camera/terrain sensing on Apple Silicon. Add a separate CUDA C++ backend when NVIDIA hardware is an actual training target.
- **Python is the experiment and compiler control plane**: PyTorch/PufferLib training, GDAL/PROJ ingestion, configuration, reports, and visualization orchestration. It must not appear in the per-environment step path.
- **FlatBuffers is the versioned host IR** for vehicle, terrain, sensor, and mission bundles. A smaller, fixed-layout `EdgeIR` remains the embedded truth for observations, actions, recurrent state, quantization, and memory.
- **MAVLink is the primary real-aircraft adapter** for PX4 and ArduPilot at bounded setpoint level. Betaflight support is secondary and must not be presented as an equivalent high-level autonomy interface.
- **MuJoCo remains an independent reference**, not the bulk trainer and not a source copied into the C implementation.

This is deliberately not a single-language system. A shared C ABI, shared typed bundles, and differential tests provide portability; forcing one source language across Metal, CUDA, flight controllers, and FPGA would provide superficial uniformity at the cost of weaker native toolchains and a larger validation gap.

## Why this is the right boundary

FlightRL already has the correct architectural seed:

- `src/flightrl/native/` is a compiled C vector environment and CPython binding.
- `src/flightrl/sixdof/` is the readable Python model and experiment lane.
- `src/flightrl/mujoco/` is an independent rigid-body/contact reference.
- the edge lane already defines byte-level observation/action contracts and treats the onboard flight controller as estimator, stabilizer, mixer, watchdog, and ultimate safety authority.

Do not replace that split. Generalize it around an explicit `flightrl_core` C ABI and generated bundle readers. The C scalar path is the executable specification; SIMD, Metal, CUDA, CMSIS-NN, and FPGA are replaceable backends whose outputs must be compared against it.

The recommendation optimizes for runtime performance, deterministic semantics, hardware reach, and evidence quality. Migration effort is not a deciding factor. Validation surface is: every additional backend must earn its place with a measured bottleneck and an independent parity suite.

## Language and runtime allocation

| Layer | Required technology | Use it for | Do not use it for |
|---|---|---|---|
| Semantic simulator and host ABI | C17 | scalar reference kernels, CPU batched kernels, safety, generated float/INT8 actor | scene authoring, training logic, GPU APIs |
| Accelerator implementations | C++20 behind `extern "C"` | CUDA kernels/graphs, Vitis HLS, device memory managers | defining different physics semantics |
| Apple GPU | MSL + thin Objective-C++ bridge | camera rays/rasterization, depth/segmentation, image perturbations, later measured physics kernels | portable source-of-truth kernels |
| NVIDIA GPU | CUDA C++ | resident batched sensor simulation and, if profiling warrants, physics | an unconditional dependency for local use |
| Training and compilation control | Python 3.13 | PyTorch, PufferLib, bundle compilation, evaluation, reports | per-step callbacks or runtime parsing |
| Geospatial compiler | GDAL + PROJ C/C++ libraries, Python orchestration | import, CRS transform, resampling, tiling, offline asset compilation | live runtime geodesy or GeoTIFF parsing |
| Portable visualization | WebGPU/WGSL, optionally | interactive map/debug renderer | authoritative training sensor generation |
| Embedded policy | generated C17 + target kernels | fixed float and INT8 actor, static arena, watchdog integration | dynamic graphs, heap allocation, Python |
| FPGA policy | hls4ml/Vitis HLS first; FINN for very low-bit fixed graphs | static streaming inference and board integration | arbitrary models or universal bitstreams |

CUDA's official programming model explicitly separates host code, device code, kernels, and host/device memory; it is the native NVIDIA path, not a general portability layer ([CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)). Metal similarly exposes direct graphics and compute access on Apple GPUs, with compute kernels dispatched over grids of parallel threads ([Metal](https://developer.apple.com/documentation/metal), [Metal compute passes](https://developer.apple.com/documentation/metal/compute-passes)). Use each native API where it is strongest and make the contract portable, rather than pretending the kernel source is portable.

PyTorch's MPS backend maps operations to Metal Performance Shaders Graph and tuned Metal kernels, so it is appropriate for learner execution and ordinary tensor work on the Mac ([PyTorch MPS backend](https://developer.apple.com/metal/pytorch/)). It is not sufficient as the simulator architecture: custom sensor kernels need explicit layout, synchronization, and parity control.

### Technologies deliberately not selected

- **Kokkos is not the foundation.** It supports CUDA, HIP, SYCL, OpenMP, and other execution spaces, but no Metal execution space; additionally, a Kokkos build enables only one device backend and one host backend ([execution spaces](https://kokkos.org/kokkos-core-wiki/API/core/execution_spaces.html), [configuration guide](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html)). Reconsider it only if maintaining distinct CUDA/HIP/SYCL physics backends becomes a measured cost after an NVIDIA backend exists.
- **SYCL is not the foundation.** SYCL 2020 is a standardized single-source C++ model, but the official overview notes that architecture-specific optimized kernel code may still differ and does not promise perfect performance portability; it also does not supply the required Apple Metal path ([Khronos SYCL](https://www.khronos.org/sycl/), [SYCL Registry](https://registry.khronos.org/SYCL/)).
- **NVIDIA Warp is a prototyping/oracle tool only.** Warp compiles annotated Python kernels to CPU or CUDA code, but Apple Silicon is CPU-only in Warp; its former simulation module was removed and superseded by Newton ([Warp basics](https://nvidia.github.io/warp/latest/user_guide/basics.html), [Warp FAQ](https://nvidia.github.io/warp/latest/user_guide/faq.html), [simulation-module transition](https://github.com/NVIDIA/warp/discussions/1067)). Do not build FlightRL's stable simulator contract on it.
- **Rust is not added to the core.** `wgpu` is a strong cross-platform graphics API with Metal, Vulkan, and DirectX 12 backends, but it does not improve the existing C/embedded boundary or the CUDA/HLS toolchains enough to justify another FFI and parity surface ([wgpu repository](https://github.com/gfx-rs/wgpu)). It remains reasonable for a future standalone ground application.
- **Dawn/WebGPU is visualization infrastructure, not the hot-path authority.** Dawn provides a native WebGPU implementation over Metal, Vulkan, D3D12, and OpenGL and is a good route to a portable viewer ([Dawn repository](https://github.com/google/dawn)). Its extra abstraction is not justified for the first learner-camera backend.

## Target architecture

```text
authoring sources                         immutable compiled bundles
TOML/JSON, URDF/SDF, CAD, GeoTIFF  ──►  VehicleBundle
mission DSL, weather distributions       TerrainBundle
calibration and thrust-stand data        MissionBundle
                                         SensorBundle
                                                │
                    ┌───────────────────────────┼───────────────────────────┐
                    ▼                           ▼                           ▼
             C17 scalar oracle          C17 SIMD CPU batch         Metal / CUDA sensors
                    │                           │                           │
                    └──────────── tensor ABI / DLPack ─────────────────────┘
                                                │
                                   PyTorch + PufferLib training
                                                │
                              PolicyBundle + restricted EdgeIR
                                   │             │              │
                              float C         INT8 C        HLS / FPGA
                                   └──────── flight-controller adapters ───► aircraft
```

The DLPack specification defines a minimal in-memory tensor structure and a C ABI across CPU and accelerator device types; use it for zero-copy producer/consumer exchange where the selected frameworks support the needed device ([DLPack specification](https://dmlc.github.io/dlpack/latest/)). Do not expose Python-owned temporary buffers through the step ABI.

## 1. Typed, versioned compilation boundary

Human-authored configuration is not runtime data. Build an offline compiler that validates and emits immutable bundles:

- `VehicleBundle`: frame geometry references; SI mass, center of mass, inertia tensor and reference frame; rotor locations/axes/directions; motor, propeller, ESC and battery models; actuator lag; sensor intrinsics/extrinsics; autopilot capabilities; uncertainty distributions; calibration provenance.
- `TerrainBundle`: source CRS and hashes; one local metric ENU origin; heightfield; collision SDF or mesh; render mesh/textures; semantic layers; weather bases and validity bounds.
- `MissionBundle`: task graph; success, abort and geofence constraints; allowed sensing and communication; payload roles; swarm size and failure model.
- `SensorBundle`: modality, sample schedule, latency/jitter, noise/dropout, rolling/global shutter, exposure, codec, extrinsics, and clock domain.
- `PolicyBundle`: observation/action schema hashes, embodiment descriptor schema, graph/weights, recurrent-state layout, calibration ranges, support envelope, and evidence identity.

Use FlatBuffers for these host-side bundles. It generates readers for C++, Python, Rust, and other languages and is designed for data readable across language and schema versions; `flatc --conform` checks whether a new schema violates the permitted evolution rules ([FlatBuffers](https://github.com/google/flatbuffers/blob/master/README.md), [`flatc` schema conformance](https://github.com/google/flatbuffers/blob/master/docs/source/flatc.md)). Commit schemas and generated compatibility fixtures, not hand-coded duplicate structs.

Do **not** use FlatBuffers directly on a tiny flight-controller loop. Generate a fixed C deployment ABI from `PolicyBundle`: byte offsets, dimensions, scaling, units, frames, recurrent reset behavior, freshness, and CRC/version fields. This preserves the strong byte-contract approach already present in FlightRL.

ONNX is useful only as an interchange checkpoint because its IR provides versioned graph, type, operator, and opset concepts ([ONNX IR specification](https://onnx.ai/onnx/repo-docs/IR.html)). It must not be the final embedded semantic authority: lower ONNX into a deliberately small `EdgeIR` with an allowlisted operator set, fixed tensor shapes, explicit accumulator widths, explicit scale/zero-point, saturation, rounding, static memory offsets, and test vectors.

## 2. Physics and batch execution

Keep the current C engine and refactor its data layout, not its language:

1. The scalar C implementation defines one environment step.
2. Store batched state in structure-of-arrays form with aligned, contiguous buffers and fixed strides.
3. Keep unit, axis, frame, quaternion, contact, actuator, and reset semantics explicit in headers and generated schema constants.
4. Add compiler-vectorized loops first; introduce architecture-specific SIMD kernels only for measured misses.
5. Keep state resident across steps. The ABI receives batch-level action/tensor views, never calls Python per environment, and returns tensor views without repacking.
6. Give each episode/environment an independent counter-based random stream so results do not depend on worker scheduling.

Physics should remain on CPU until profiling shows that it limits end-to-end samples per second after camera sensing and learner time are included. A GPU physics port is justified only if it keeps full environment state resident and produces a material end-to-end gain; PCIe or unified-memory traffic must be included in the measurement.

PufferLib's native backend is explicitly designed for vectorized environments, and its Ocean environments demonstrate C implementations intended for very high local throughput ([PufferLib docs](https://puffer.ai/docs.html), [Ocean environments](https://puffer.ai/ocean.html)). Preserve that lane rather than inserting a general robotics middleware into training.

MuJoCo stays independently specified. Compare force-free trajectories, hover trim, thrust/torque responses, actuator transients, contacts, and sensor geometry under declared tolerance envelopes. Do not chase identical contact traces: independent engines should agree on invariants and mission-relevant envelopes, not floating-point accidentals.

## 3. Terrain, camera, and other sensors

GDAL provides common raster/vector data models and C, C++, and Python APIs; PROJ performs coordinate-reference-system transformations ([GDAL](https://gdal.org/en/stable/), [GDAL raster model](https://gdal.org/en/stable/user/raster_data_model.html), [GDAL vector model](https://gdal.org/en/stable/user/vector_data_model.html), [PROJ usage](https://proj.org/en/stable/usage/index.html)). Use their Python bindings for orchestration because the heavy operations already execute in native libraries.

The offline terrain compiler should:

1. identify source CRS, vertical datum, resolution, license, and provenance;
2. transform a bounded operating region to a local metric ENU frame;
3. produce separate collision, sensing, and display levels of detail;
4. precompute tiles, normals, semantic masks, and optional SDF/occupancy structures;
5. hash every source and transformation parameter;
6. fail on missing CRS, ambiguous altitude reference, or invalid coverage.

The simulator must never parse GeoTIFF, online map APIs, or geodetic coordinates in the hot loop.

Build two renderers with different jobs:

- **sensor renderer:** deterministic low-resolution grayscale/RGB, depth, segmentation, optical-flow/event approximations, and range rays using exactly the deployed intrinsics, shutter, latency, sample/hold, and image packing;
- **display renderer:** attractive live terrain, drone, planned path, uncertainty, and camera overlays for demos.

Start the sensor renderer as batched Metal kernels over compiled heightfields/meshes. Keep device state and images resident through learning when possible. Port the same contract independently to CUDA for NVIDIA training. Validate rays, projections, occlusion, depth convention, image packing, and temporal sampling against CPU golden scenes; do not require bitwise equality for floating GPU paths.

Photometric randomization must cover a declared camera envelope—exposure, blur, rolling shutter, noise, compression, illumination, texture and weather—while geometric truth stays separately inspectable. Pretty rendering must never silently alter training observations.

## 4. Cross-airframe and swarm policy design

Do not claim one opaque policy is universal. Train an **embodiment-conditioned policy within a measured support envelope**. Supply either a compact descriptor or an online-identified latent containing normalized quantities such as mass, inertia, thrust-to-weight, arm geometry, actuator lag, drag basis, sensor extrinsics/latency, battery state, and available action limits.

Universal Policies with online system identification showed the relevant pattern: a policy can adapt using an inferred dynamics embedding when trained across varying physical parameters ([Yu et al., RSS 2017](https://www.roboticsproceedings.org/rss13/p48.pdf)). Dynamics randomization likewise supports transfer by training across simulator dynamics rather than one nominal model ([Peng et al., 2018](https://arxiv.org/abs/1710.06537)). These papers justify the method, not a blanket generalization claim.

Promotion requires held-out evaluation across:

- entire vehicle families, not just new parameter seeds;
- mass/inertia, motor/propeller, battery, latency and damaged-actuator corners;
- new terrain regions, wind fields, lighting, camera calibration and packet loss;
- swarm sizes and communication graphs not used for gradient updates.

For swarm learning, use centralized training with decentralized execution. Each deployed actor receives local state/perception, neighbor messages or relative tracks, mission context, and embodiment descriptor; a privileged critic may see global truth only during training. The communication model must simulate bandwidth, delay, clock offset, loss, stale neighbors and identity changes.

The deployable default action remains bounded velocity plus yaw-rate/setpoint intent. The flight controller owns attitude/rate stabilization and mixing. A separate aggressive-flight policy may emit body rates plus collective thrust, but it is a different capability profile with tighter latency, identification and safety evidence; it must not silently share the portable-policy claim.

## 5. Flight-controller integration

Define one `AutopilotAdapter` capability contract with time synchronization, state validity, command mode, arming state, watchdog, geofence, manual takeover, and failsafe reporting.

- **PX4:** use MAVLink or ROS 2 offboard setpoints. PX4 accepts position, velocity, acceleration, attitude, body-rate and thrust forms, requires a continuous offboard proof-of-life above 2 Hz, and exits according to configured loss behavior when the stream stops ([PX4 Offboard Mode](https://docs.px4.io/main/en/flight_modes/offboard)).
- **ArduPilot:** use MAVLink Guided mode with `SET_POSITION_TARGET_LOCAL_NED`, global position targets, or `SET_ATTITUDE_TARGET`; velocity/acceleration commands must be refreshed and the vehicle stops after the documented timeout ([ArduCopter Guided commands](https://ardupilot.org/dev/docs/copter-commands-in-guided-mode.html)). The common MAVLink message defines frame, type mask, position, velocity, acceleration, yaw, and yaw-rate fields ([MAVLink message](https://mavlink.io/en/messages/common.html#SET_POSITION_TARGET_LOCAL_NED)).
- **Betaflight:** support it through a distinct adapter. MSP is a request/response command protocol with API-version negotiation ([Betaflight MSP reference](https://betaflight.com/docs/development/MSP-Protocol-Reference-Dev)); it is not an equivalent documented high-level autonomy contract. Use an FC-supported serial control path only after bench-testing latency, loss behavior, arming, mode switching and manual takeover for the exact firmware target.

Neither Wi-Fi nor an ESP32 should be the primary control/safety link. They can carry payload data or low-rate coordination. The autopilot link must have an independently tested loss-of-link response, and the RC/manual path must remain capable of immediate intervention during development.

## 6. Training and export

Use PyTorch for the learner and PufferLib for high-throughput collection. `torch.compile` is a JIT optimization facility; `torch.export` is the AOT graph artifact route and explicitly separates export from deployment/runtime concerns ([`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile), [`torch.export`](https://docs.pytorch.org/docs/stable/export.html)). Therefore:

1. train with ordinary PyTorch modules;
2. export a frozen graph with fixed recurrent and tensor contracts;
3. validate and lower to `EdgeIR`;
4. generate float C as the deployment oracle;
5. calibrate/quantize and generate INT8 C;
6. substitute target kernels only after golden-sequence parity.

Use a static arena and forbid runtime allocation in the actor. Every tensor has a compile-time maximum shape. Record per-layer accumulator ranges, saturation counts, recurrent-state reset vectors, worst-case execution time, stack/arena sizes, compiler flags and target identity.

For Cortex-M targets, CMSIS-NN supplies optimized neural-network kernels aligned with the TensorFlow Lite Micro INT8/INT16 quantization specification ([CMSIS-NN repository](https://github.com/ARM-software/CMSIS-NN)). Use these behind generated `EdgeIR` calls where the exact operator semantics match. ExecuTorch has a small C++ runtime, planned memory, operator registration and platform backends, so it is appropriate only if model variability becomes more valuable than the smaller auditable generated-C runtime ([ExecuTorch runtime overview](https://docs.pytorch.org/executorch/stable/runtime-overview.html)).

### FPGA decision

Prototype the current compact recurrent actor with **hls4ml plus Vitis HLS**, because hls4ml exposes recurrent/PyTorch support while Vitis supplies C simulation, synthesis, C/RTL co-simulation and IP/kernel export ([hls4ml support status](https://github.com/fastmachinelearning/hls4ml/blob/main/docs/intro/status.rst), [hls4ml FAQ](https://fastmachinelearning.org/hls4ml/intro/faq.html), [Vitis HLS flow](https://docs.amd.com/r/en-US/Vitis-Tutorials-Getting-Started/Using-the-New-Vitis-Unified-IDE)). Move very-low-bit fixed CNN/MLP sections to FINN only when the topology and quantization justify a streaming dataflow design. FINN targets quantized networks but supports a subset of ONNX/QONNX and expects transformation or custom work for unsupported nodes ([FINN overview](https://finn.readthedocs.io/en/latest/), [FINN getting started](https://finn.readthedocs.io/en/latest/getting_started.html), [FINN FAQ](https://finn.readthedocs.io/en/latest/faq.html)).

Vitis HLS C/C++ is a synthesis subset without ordinary dynamic allocation or operating-system behavior, reinforcing the static `EdgeIR` requirement ([Vitis HLS C functions](https://docs.amd.com/r/en-US/ug1399-vitis-hls/Coding-C/C-Functions)). Handwritten RTL should be limited to narrow timing, protocol or safety blocks whose behavior is clearer than an HLS implementation.

One FPGA **board** can serve many drones by loading different signed vehicle calibration and policy bundles. One FPGA **bitstream** can serve a family only while tensor shapes, operator graph, precision and I/O schedule remain within its compiled limits. Do not sell “one universal bitstream”; sell a stable hardware/runtime interface plus a compiler and verified family-specific artifacts.

## 7. Validation and evidence

Each optimization backend must pass the same layered gates:

1. **Schema tests:** old/new bundle compatibility, rejected unknown required fields, unit/frame checks, deterministic canonical hashes.
2. **Kernel golden vectors:** arithmetic edge cases, quaternion normalization, saturation, contact, sensor projection, image packing and reset.
3. **Differential one-step tests:** scalar C versus SIMD/Metal/CUDA at fixed state/action/RNG inputs.
4. **Trajectory envelopes:** accumulated position, attitude, energy, collision and task differences over long rollouts.
5. **Independent model checks:** C versus MuJoCo on analytically constrained experiments and system-identification data.
6. **Policy sequence parity:** complete recurrent sequences, including resets, dropouts and stale observations; float C versus PyTorch tolerance, INT8 host versus target bit-exact.
7. **Property tests:** finite state, energy/no-thrust decay expectations, bounded action, invariant frames, monotonic clocks, no stale-command acceptance.
8. **Performance tests:** throughput and p50/p95/p99 step/sensor/actor latency, memory traffic, peak resident memory, thermal state and compiler/device identity.
9. **Hardware gates:** shadow, tether/cage, single-axis/low-energy, bounded flight, then expanded envelope; simulator success never authorizes flight.

Floating GPU backends need tolerance-based component and trajectory evidence, not bitwise parity. Fixed-point deployment needs bit-exact parity because rounding, saturation and accumulator width are part of `EdgeIR`. Store minimal failing seeds and compiled bundle hashes so backend disagreements are replayable.

## 8. Concrete mapping and build order

### Preserve now

- `src/flightrl/native/`: make it `flightrl_core` semantic authority; preserve the CPython interface while extracting a stable C ABI.
- `src/flightrl/sixdof/`: readable model, scenario compiler front end, diagnostics and research experiments.
- `src/flightrl/mujoco/`: independent oracle and contact/sensor validation.
- existing edge observation/action/wire contracts: evolve them into generated `EdgeIR`, retaining current sequence-parity gates.
- existing bounded setpoint policy: make it the default capability exposed by MAVLink adapters.

### Add in this order

1. **Contract freeze:** SI units/frames, vehicle/terrain/mission/sensor schemas, scalar C golden corpus, embodiment descriptor, capability matrix.
2. **Offline compiler:** GDAL/PROJ ingestion and FlatBuffer bundles; no runtime map parsing.
3. **CPU data path:** SoA batch state, aligned tensor-view ABI, benchmark without Python callbacks or copies.
4. **Metal sensor backend:** low-resolution camera/depth/segmentation and perturbations; CPU golden scenes; learner handoff without host round trips.
5. **Cross-airframe training:** declared randomization envelope, embodiment conditioning or online ID, held-out family tests, decentralized swarm actor.
6. **MAVLink adapter:** PX4 SITL and ArduPilot SITL first, then hardware-in-loop and bounded real flight.
7. **Export compiler:** PyTorch to `EdgeIR` to generated float C and INT8 C; full recurrent parity and static memory proof.
8. **CUDA backend:** only when NVIDIA scale is available and a profiler shows the expected end-to-end win.
9. **FPGA proof:** hls4ml/Vitis for the frozen recurrent graph; FINN only for qualifying low-bit subgraphs.

### Kill criteria

- Reject a backend if it cannot beat the simpler path end to end after data movement is counted.
- Reject a cross-airframe claim if held-out vehicle families fail even when in-distribution parameter sweeps pass.
- Reject a renderer if display-quality changes alter learner observations without a versioned sensor-contract change.
- Reject an embedded artifact without golden recurrent sequences, static memory bounds and worst-case latency evidence.
- Reject an autopilot integration without command-timeout, manual-takeover and loss-of-link tests on the exact firmware.

## Bottom line

The defensible product is not “a universal policy” or “one FPGA that controls every drone.” It is a local-first compiler-and-runtime stack that turns typed vehicle, terrain, sensor and mission descriptions into fast simulation, explicitly bounded policies, and reproducible target artifacts. The valuable generalization boundary is the versioned contract plus the evidence envelope. FlightRL's current native C + Python + MuJoCo split is already the right nucleus; deepen it with FlatBuffer compilation, native Metal sensing, a C/INT8 deployment compiler, and capability-driven MAVLink adapters before adding another portability framework or rewriting the core.
