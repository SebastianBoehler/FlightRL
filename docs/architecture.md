# Architecture

## Product boundary

FlightRL is a local-first mission-to-machine research stack. Its intended
output is not one universal motor policy. It compiles explicit vehicle,
terrain, sensor, and mission descriptions into:

- high-throughput simulation inputs;
- policies with a measured embodiment support envelope;
- reproducible desktop, embedded-C, and FPGA deployment artifacts;
- evidence binding the source, contracts, calibration, and target runtime.

The current implementation is still centered on a Crazyflie edge target and a
small set of desktop environments. New modules must move toward the target
architecture below without weakening the current fail-closed hardware gates.

## System shape

```text
authoring inputs                 immutable compiled bundles
terrain / CAD / calibration --> vehicle / terrain / sensor / mission
                                       |
                  +--------------------+--------------------+
                  v                    v                    v
             scalar C core       optimized backends    MuJoCo reference
                  +--------------------+--------------------+
                                       |
                              Python / PyTorch learner
                                       |
                         policy bundle + restricted EdgeIR
                              /          |          \
                         float C       int8 C      HLS / FPGA
                              \          |          /
                             autopilot adapters and safety
```

Generalization comes from stable contracts and replaceable adapters. It does
not come from forcing all implementations into one language or silently
reinterpreting same-shaped arrays.

## Language allocation

- **C17** owns scalar numerical semantics, deterministic batch stepping,
  safety-critical helpers, and generated embedded actors.
- **C++20** may implement CUDA, HLS, and vendor-host adapters behind the C
  interface. It does not define different physics semantics.
- **Python** owns experiment orchestration, PyTorch/PufferLib training,
  offline compilation, evaluation, and reports. It stays out of per-environment
  hot loops.
- **Metal** is the first Apple GPU sensor backend. **CUDA** is added when an
  NVIDIA target and an end-to-end bottleneck justify it.
- **WebGPU/WGSL** may power the portable interactive viewer, never the
  authoritative sensor implementation.

Porting effort is not an architectural constraint. Validation surface is: a
new implementation is accepted only when it produces a measured benefit and
passes the reference parity gates.

## Core seams

### Native core

`native/flightrl_core.h` is the versioned host interface for contiguous
six-DoF batches. The CPython binding is one adapter at this seam. Future tensor,
Metal, CUDA, or standalone hosts must use the same units and state/action
semantics rather than call implementation functions directly.

The scalar C path is the executable specification. Optimized backends are
replaceable implementations and require differential one-step and trajectory
tests. MuJoCo stays independently specified for rigid-body, contact, and sensor
validation.

### Artifact identity

`flightrl.artifact_identity` owns canonical JSON encoding, SHA-256 payload
binding, and file identity. Evidence contracts with these semantics use this
interface rather than defining local encoders. Identity-bearing JSON rejects
non-finite values; domain-specific framed tensor digests remain separate.

Host bundles will eventually use FlatBuffers and aligned numeric chunks. JSON
continues to be appropriate for human-readable evidence reports, but is never
parsed in a simulation or control hot loop.

### Embodiment

`flightrl.sixdof.embodiment` defines the first explicit physical descriptor for
the current rate-lag six-DoF model. `SixDofEnv.embodiment_descriptors()` exposes
the descriptor actually sampled for each environment.

This descriptor is not yet a complete vehicle bundle: the current simplified
physics does not model arbitrary rotor topology, full inertia, materials, or
motor/propeller maps. Those fields must be added only together with a simulator
implementation and validation data that consume them.

### Scenario bundle

`compile_scenario_bundle(...)` is the first offline compiler seam. It consumes
the repository's validated physics, room, sensor, and resolved-mission types
and emits fixed little-endian float32 arrays plus one canonical manifest. The
manifest binds array shapes, field ordering, coordinate frames, content
digests, and explicitly simulation-only authority.

`write_scenario_bundle(...)` never overwrites an existing directory, and
`load_scenario_bundle(...)` revalidates the manifest and every array digest.
This is the common input contract for later native CPU, Metal, CUDA, and SITL
adapters; it is not a physical-flight deployment bundle.

### Policy and autopilot

The default portable policy emits bounded velocity, yaw-rate, or trajectory
intent. PX4/ArduPilot or the Crazyflie STM32 owns estimation, attitude/rate
stabilization, mixing, watchdogs, and motor authority. An aggressive body-rate
policy is a separate capability profile with tighter timing and identification
evidence.

The current `aideck-navigation-policy-v3` actor remains the only concrete edge
target. It consumes one 64x48 gray4 frame, 19 normalized telemetry values, a
closed-vocabulary target ID, and recurrent state. Its proposals cannot assert
mission completion or grant flight authority.

## Current implementation boundary

The repository does not yet contain the full target stack. In particular:

- the native core models fixed quadrotor six-DoF state, not arbitrary rotor
  topology, deformable frames, or material behavior;
- the edge-v3 PyTorch actor is a design reference, not a frozen float-C, int8,
  GAP8, or FPGA implementation;
- no Metal or CUDA camera/sensor backend exists;
- no multi-aircraft environment, network model, decentralized swarm policy, or
  held-out-airframe promotion gate exists;
- no PX4 or ArduPilot adapter implements this policy contract;
- no typed deployment bundle grants learned physical-flight authority.

Existing hardware modules support Crazyflie connection, preflight, capture,
telemetry, nonlearned bring-up, and evidence collection. The generic
sim-to-real manifest intentionally emits no hardware-approved learned
checkpoint. These are retained safety boundaries, not compatibility gaps to
work around.

## Desktop environments

### Native C

The extension owns contiguous vector state and writes observations, rewards,
termination state, and metrics in place. `SixDofEnv` is intentionally named for
its model rather than one aircraft. Crazyflie and Puffer values are explicit
physics profiles, not simulator identities.

The fixed-door environment remains a privileged teacher. Its observation and
mission contracts do not become edge-v3 merely because tensor dimensions are
compatible.

### MuJoCo

MuJoCo is the independent, slower reference for dynamics, contacts, geometry,
sensor semantics, and model-calibration experiments. It is not the bulk
trainer and its implementation must not be copied into the C reference.

### Training

PyTorch owns learners and export-time graphs. PufferLib consumes the native
environment through generated Ocean adapters. State should remain resident
across rollout steps; Python callbacks and repacking are excluded from the hot
path. A future tensor adapter should use the native core interface and DLPack
where zero-copy ownership can be proven.

## Cross-airframe and swarm contract

One runtime can support many aircraft. A shared policy is promoted only inside
a measured support envelope.

Cross-airframe training conditions the actor on physical descriptors or an
identified dynamics latent. Evaluation holds out entire vehicle families as
well as parameter seeds. A failure on a held-out family rejects the shared
policy claim; the same compiler may still produce a family-specific policy.

Swarm policies use centralized training and decentralized execution. Each
actor sees local sensing, mission context, embodiment, and bounded neighbor
messages. Communication simulation must include latency, loss, stale data,
clock offset, and changing neighbor identity.

## Deployment contract

ONNX may be an interchange checkpoint. A restricted `EdgeIR` is the deployment
authority and must specify:

- fixed tensor shapes and an operator allowlist;
- static memory offsets and maximum arena size;
- accumulator widths, rounding, saturation, and quantization scales;
- recurrent-state layout and reset rules;
- units, coordinate frames, timestamps, and freshness requirements;
- test vectors and a content identity.

The first executable target chain is PyTorch -> float C -> int8 C. FPGA work
starts only after this chain is frozen and bit-exact across recurrent
sequences. hls4ml/Vitis is the first recurrent-policy route; FINN is reserved
for compatible low-bit subgraphs.

## Evidence gates

Every new implementation must pass the applicable layers:

1. schema, units, frame, and canonical-identity validation;
2. scalar kernel golden vectors;
3. differential backend one-step tests;
4. long-rollout trajectory envelopes;
5. independent MuJoCo or measured-hardware checks;
6. complete recurrent-sequence export parity;
7. throughput, p50/p95/p99 latency, memory, energy, and data-movement reports;
8. staged hardware gates with independent timeout and manual takeover.

Floating accelerator paths use declared numerical tolerances. Fixed-point
deployment is bit-exact because rounding and saturation are contract semantics.

## Implementation order

1. Extend the initial native ABI, artifact identity, embodiment, coordinate
   frame, and scenario-bundle contracts as real backends require new fields.
2. Build the offline terrain and vehicle compiler; runtime map parsing is
   forbidden.
3. Keep the CPU physics reference fast while adding deterministic Metal camera,
   depth, segmentation, and perturbation kernels.
4. Add embodiment-conditioned policies and held-out-airframe evaluation.
5. Add PX4 and ArduPilot SITL adapters before physical integrations.
6. Freeze `EdgeIR`, generate float/int8 C, and prove sequence parity.
7. Add CUDA and FPGA implementations only against stable contracts and measured
   bottlenecks.

Unsupported paths fail closed. A successful simulation, teacher, export, or
replay report never authorizes physical flight.
