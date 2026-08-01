# Semantic mission architecture

## Goal

Support bounded commands such as:

```text
Fly to the desk corner, hold for 3 seconds, then go to the door.
```

The language layer must not directly command motors. It compiles text into a
small validated contract that both simulation backends and the Crazyflie
setpoint path can consume:

```text
text command
  -> MissionPlan
  -> semantic target resolver
  -> fixed numeric mission rows
  -> visual navigation / avoidance policy
  -> body velocity, vertical velocity, and yaw-rate setpoints
  -> Crazyflie firmware stabilizer
  -> motors
```

This is a small VLA or VLN-lite architecture. It is not an open-vocabulary VLA
and it is not an LLM-to-motor controller.

## Implemented slice

The first implementation is in:

- `src/flightrl/navigation/mission_spec.py`
- `src/flightrl/navigation/mission_compiler.py`
- `src/flightrl/navigation/semantic_scene.py`
- `src/flightrl/mujoco/semantic_scene.py`

It provides:

- a constrained parser for `go/fly/navigate/move to`, `hold`, `land`, and
  `abort`;
- semantic objects with unique names, aliases, 3D bounds, collision flags, and
  explicit approach poses;
- safe corner target generation outside collidable objects;
- deterministic failures for unknown language, unknown objects, ambiguous
  aliases, and unsafe targets;
- MuJoCo geometry injection using the same object definitions;
- a stable versioned row contract for the future native Ocean mission buffer.

Each resolved mission step contains:

| Index | Field |
| ---: | --- |
| 0 | command id |
| 1 | semantic object index, or `-1` |
| 2 | target anchor id |
| 3-5 | target x, y, z in meters |
| 6 | target yaw in radians |
| 7 | step duration in seconds |
| 8 | speed scale |

The text parser is intentionally narrow. A host-side LLM can later translate
broader language into the same schema, but the schema validator and safety
stack remain authoritative.

## What MuJoCo is doing

MuJoCo is a native physics and rendering engine with official Python bindings.
Python constructs an MJCF/XML model, owns experiment orchestration, and calls
the native engine through `mujoco.MjModel`, `mujoco.MjData`, `mujoco.mj_step`,
and `mujoco.Renderer`.

In FlightRL, `MuJoCoCrazyflieEnv` currently creates one `MjData` per simulated
drone and loops over them in Python. The rigid-body integration, contacts, and
rendering are native MuJoCo operations, but Python still schedules each
environment and frame. This makes the backend useful for:

- checking geometry, contacts, and camera pose;
- rendering recognizable desk, door, wall, and obstacle scenes;
- validating that the target resolver and camera observe the same world;
- comparing native C dynamics and images against an independent engine.

It is not the intended high-throughput training backend.

Official references:

- https://mujoco.readthedocs.io/en/stable/python.html
- https://mujoco.readthedocs.io/en/stable/XMLreference.html

## What Ocean and PufferLib are doing

Ocean is the native environment side of PufferLib. FlightRL exports its C
environment modules plus a generated `binding.c` into a PufferLib checkout,
then compiles them into PufferLib's native extension.

The resulting hot path is:

```text
PuffeRL Python/PyTorch trainer
  -> contiguous action buffer
  -> compiled Ocean C vec_step
  -> C physics + C scene raycaster + C reward/task logic
  -> in-place observation/reward/done buffers
  -> PyTorch CNN/MinGRU policy
```

Python still configures experiments, runs PPO, logs metrics, and saves
checkpoints. It does not step each Ocean environment or render each pixel.
That distinction is why the current native scene can step at roughly 169k
agent-steps/s while the Python MuJoCo renderer is a correctness reference.

PufferLib reference:

- https://github.com/PufferAI/PufferLib

## Current setup versus semantic target training

The current native visual environment renders a room and one generic obstacle.
Its policy input is:

```text
64x48 gray4 appearance
signed frame delta
motion mask
body-frame target direction, distance, and yaw error
```

That six-value target vector is privileged information: it tells the student
where the target already is. It is useful for waypoint following and visual
obstacle avoidance, but it does not prove that the model can find a desk or
door from language and pixels.

The target architecture must generalize to unseen rooms without manually
configured object coordinates:

```text
object phrase such as "sink" or "door"
  -> open-vocabulary grounder on a higher-resolution camera frame
  -> target box, mask/heatmap, confidence, and time-since-seen
  -> category-agnostic Puffer navigation policy at the control rate
  -> search, approach, reacquire, avoid, hold, or safe abort setpoints
```

The grounder is responsible for what the requested object looks like. The
navigation policy is responsible for how to safely reach any grounded target.
This avoids relearning flight behavior separately for every object category.

The first grounder can run offboard because semantic grounding does not require
the 65 Hz control rate. The GAP8 path can later receive a distilled
closed-vocabulary detector or tracker, while the same navigation policy and
target-observation contract remain unchanged.

The Mac-hosted Grounding DINO path is development scaffolding, not the intended
final runtime. The primary deployment design is one shared target-conditioned
recurrent policy:

```text
validated mission or prompt
  -> compact target token
  -> on-edge detector/tracker and recurrent visual navigation policy
  -> bounded velocity and yaw-rate setpoints
  -> Crazyflie firmware stabilizer
```

Free-form language may initially be compiled to the target token on the host at
mission start; it does not need to remain in the control loop. A checkpoint
compiled with one constant token can be useful for a small “always find the
monitor” deployment test, but training one separate navigation policy per
object is not the target architecture because it duplicates exploration,
avoidance, and approach behavior.

The current simulator vocabulary is `door`, `monitor`, and `sink`. Adding
`window` requires a scene object, command-token entry, detector examples, and
held-out evaluation; it is not implemented by the current checkpoint.

A privileged simulation teacher may see object poses and labels. The deployed
student must see only camera data, IMU/odometry, previous action, mission state,
and the grounder's relative target signal. Ground-truth target direction must
not be present in the deployed student observation.

A learned neural world model is not required. Explicit three-dimensional
scenes, semantic assets, camera rendering, physics, and domain randomization
already generate closed-loop observations. A learned world model becomes
interesting later for prediction, planning from real logs, or reducing
rendering cost.

## Implementation plan

### 1. Mission execution parity

- Add a small mission cursor that activates one resolved row at a time.
- Feed the active target into `MuJoCoVisionPufferEnv`.
- Advance `go-to -> hold -> next target -> land` only through explicit success
  events and timeout/abort gates.
- Verify a scripted firmware-style waypoint baseline before training.

Expected effort: about one engineering day.

### 2. Native semantic Ocean scenes

- Replace the single native obstacle with fixed-capacity arrays of labeled box
  primitives.
- Give each object geometry, material/texture seed, collision state, semantic
  id, and approach anchors.
- Randomize room dimensions, object layout, lighting proxy, textures, exposure,
  and distractors per episode.
- Load resolved mission rows once at reset and keep the active row in C state.
- Preserve the current `64x48` gray4 observation contract.

Expected effort: two to four engineering days.

### 3. Teacher and student contracts

- Privileged teacher observation: drone state, target pose, object poses,
  collision geometry, and mission phase.
- Visual student observation: high-rate image channels, IMU/odometry, previous
  action, mission phase, target heatmap/box, confidence, and target staleness.
- Semantic grounder input: a higher-resolution camera frame and the object
  phrase; output must use the same target-observation contract for every class.
- Output: residual body velocity, world vertical velocity, and yaw rate around
  the safe waypoint controller.
- Rewards: sparse mission completion plus progress potential, collision and
  clearance costs, hold stability, smooth actions, and timeout.

Expected effort: two to five engineering days before meaningful sweeps.

### 4. Causal evaluation gates

- Compare full camera against masked/shuffled camera.
- Hold out complete rooms, object instances, layouts, textures, lighting, and
  command wording.
- Require evaluation without manually configured object coordinates.
- Measure target grounding precision/recall on real AI Deck frames before
  allowing the grounder to command navigation.
- Require multi-step completion, collision rate, minimum clearance, hold error,
  and action smoothness.
- Reject any checkpoint without a material full-camera advantage.

The current visual residual checkpoint fails this gate and remains
simulation-only.

### 5. Real-room progression

- Validate an offboard open-vocabulary grounder on recorded full-resolution AI
  Deck frames for multiple object classes and unseen instances.
- Replay synchronized frames, telemetry, target boxes, confidence, staleness,
  and mission state through the navigation policy.
- Run policy shadow mode while the firmware controller flies.
- Run tethered or bounded setpoint trials only after replay and shadow gates.
- Test in a room excluded from training without entering object coordinates.
- Treat an absent or low-confidence target as search-then-hold/abort, never as a
  guessed waypoint.

The first real semantic demo must therefore identify and approach requested
objects in an unseen room. A room-specific coordinate map is not an acceptable
substitute for visual grounding.

## Mac-hosted semantic discovery prototype

The first open-vocabulary test path is implemented independently of
room-specific simulation target coordinates:

- `src/flightrl/semantic/grounding_dino.py` grounds a text prompt in AI Deck
  monochrome frames with Grounding DINO on the Mac.
- `src/flightrl/semantic/controller.py` implements target scan, visual yaw
  alignment, hold, timeout, and an explicitly gated bounded-reposition state.
- `src/flightrl/semantic/worker.py` keeps camera capture independent from the
  slower semantic model and always processes the latest frame.
- `src/flightrl/semantic/dataset.py` records raw and annotated frames,
  detections, telemetry, and the command associated with each processed frame.
- `scripts/evaluate_aideck_grounding.py` runs the same detector contract on
  archived data.
- `scripts/crazyflie_semantic_find.py` runs camera-only collection by default
  and requires separate confirmations for takeoff, semantic yaw control, and
  translational exploration.

The grounder evaluates the target together with common room distractors and
retains exact target labels only. This matters at the AI Deck's `162x122`
semantic-stream resolution: a target-only `door` prompt produced false
positives on monitor rectangles, while the distractor-aware contract rejected
that negative scene and retained monitor detections in five of five positive
frames.

The default first flight begins with a fixed-position 360-degree yaw scan at 20
degrees per second. Detections are logged but cannot terminate this initial
sweep. The controller then aligns the camera to a reacquired target, holds, and
lands. If an object remains occluded from that position, the controller can
move through a 0.25 m cross-pattern and scan again, but this translation
requires `--confirm-bounded-exploration`.

With only the AI Deck and Flow Deck mounted, Flow Deck odometry and firmware
stabilization do not provide forward obstacle clearance. The bounded
reposition state is implemented for later gated tests; it is not evidence of
safe arbitrary-room exploration. Visual clearance/time-to-contact must pass
replay and shadow evaluation before that mode is treated as autonomous
obstacle avoidance.

### Verified host behavior

On archived `162x122` monochrome AI Deck frames:

- `computer monitor`: five detections in five sampled frames;
- negative `door` scene: zero retained door detections after distractor
  competition;
- Apple MPS inference: approximately 305 ms median after warm-up;
- firmware stabilization remains unchanged and runs independently of the host
  semantic update rate.

The active `64x48` gray4 firmware is appropriate for a future tiny policy but
is rejected by the semantic runner. Text grounding first requires the existing
`qqvga-pipelined-65fps` JPEG profile:

```bash
AIDECK_UDP_FLASH_CONFIRM=FLASH_AIDECK_UDP \
  scripts/aideck_udp_streamer.sh flash-semantic
```

After power cycling and connecting the Mac to the AI Deck access point, the
non-flying camera gate is:

```bash
uv run --extra semantic --extra hardware \
  python scripts/crazyflie_semantic_find.py \
  --prompt "computer monitor" \
  --duration-s 10
```

Only after the current room's positive and negative captures have been checked
should the first yaw-only flight be run:

```bash
uv run --extra semantic --extra hardware \
  python scripts/crazyflie_semantic_find.py \
  --prompt "computer monitor" \
  --flight \
  --confirm-flight \
  --confirm-semantic-yaw-control
```

### First live discovery results

The first `computer monitor` flight reached 0.3 m but accepted an
appliance-shaped rectangle as a monitor after about 60 degrees of yaw. That run
is a grounding failure, not mission success.

The corrected `window` run used a mandatory 18-second scan at 0.55 m:

- 481 degrees total yaw travel;
- altitude between 0.50 and 0.62 m;
- no tumble or safety abort;
- a correctly localized window candidate at 0.82 confidence;
- mission timeout after intermittent weak reacquisition, followed by a clean
  landing.

Replay showed that a later 0.32-confidence detection covered nearly the entire
frame. The grounder now rejects implausibly large boxes, and the discovery
controller records the strongest candidate yaw during the full sweep and
explicitly returns to that bearing before visual tracking. The fixes passed the
focused controller and stream tests but have not been flown yet. Battery
rebounded to 40 percent after the window run, so another live iteration requires
charging first.

### Resolution and bit-depth ablation

`scripts/evaluate_semantic_resolution_sweep.py` evaluates matched source frames
at:

```text
324x244, 243x183, 162x122, 128x96, 96x72, 64x48, 48x36
```

Each spatial resolution is tested at 8-bit and 4-bit grayscale. The report
compares each variant with the QVGA 8-bit baseline using frame-level target
recall, new false positives, confidence, and box IoU. Raw, degraded, and
annotated frames are retained for inspection.

A preliminary ten-frame screen sweep is stored in
`artifacts/semantic/screen-resolution-sweep-qvga-10frame-20260726/`. The
screens are large and close in this archive, so this is a pipeline validation,
not the final operating-distance result:

- QVGA through `128x96`, at both bit depths, retained all detections with at
  least 0.97 median box IoU.
- `96x72` 8-bit retained the signal, while 4-bit fell to 0.40 recall.
- `64x48` happened to retain all detections, demonstrating that model
  degradation is not monotonic under resizing.
- `48x36` continued to emit monitor detections but switched between monitor
  instances, producing zero IoU against the QVGA-selected monitor. It therefore
  fails localization stability even though class confidence remains.

The real boundary must be measured from one QVGA capture set containing the
same screen at known distances, preferably 1.0, 1.5, 2.0, 2.5, and 3.0 meters.
Only the highest-resolution stream is captured live; every lower-resolution
input is generated from the same frames. This isolates resolution and bit depth
from exposure, pose, and scene changes.

The QVGA capture profile is:

```bash
AIDECK_UDP_FLASH_CONFIRM=FLASH_AIDECK_UDP \
  scripts/aideck_udp_streamer.sh flash-semantic-highres
```

### Frame-integrity correction

The first QVGA monitor flight exposed raster-wrap seams: a sharp horizontal
boundary followed by a vertical boundary inside an otherwise decodable JPEG.
These regions are pieces of different sensor frames, not dark objects in the
room. The affected capture loop allowed both camera DMA buffers to drain while
JPEG and CPX processing ran, then queued the next transfer while the sensor was
already partway through a frame.

The GAP SDK camera contract requires at least two buffers to stay queued while
the camera is running. PRs
[#157](https://github.com/bitcraze/aideck-gap8-examples/pull/157) and
[#158](https://github.com/bitcraze/aideck-gap8-examples/pull/158) now use three
buffers: two remain queued for capture and one is reserved for the consumer.
When processing is slower than capture, complete frames are discarded.

Both corrected profiles build successfully:

- QVGA JPEG, pipelined, 60 FPS sensor timing:
  `246b1b7534483bf1238e5725dab8e55c8a27a66b2aab1ff0477784fd2e7490a5`
- `64x48` packed gray4, pipelined, 65 FPS sensor timing:
  `94cbee3a07eba2e7fb7a978f60f4c179c6d7f8bd3b8850fcaf532120cfbf9e9a`

The corrected QVGA image passed:

- 100 stationary frames at 13.71 Hz, with one dropped frame and no visible
  raster-wrap seam;
- 220 hand-moved frames at 13.77 Hz, with zero dropped frames and no visible
  seam across large yaw and pitch changes;
- monitor grounding in `13/18` sampled live frames under the final calibrated
  gate, with detections aligned to the real screen.

Do not use the earlier QVGA flight to assess grounding precision: corrupt frame
regions produced several accepted proposals.

The first no-screen view exposed a wide dark wall feature that reached CLIP
probability `0.555`. A second view initially labeled as a wall negative contains
a monitor-shaped object at the upper-left edge and is therefore not a valid
negative.

Adding desk, window, and wall as competing Grounding DINO labels suppressed the
real monitor proposals. On the same 12 positive frames, this produced `2/12`
detections, while target-only proposals produced `10/12`. The final camera and
replay gate therefore uses:

- target-only Grounding DINO proposals with confidence at least `0.25`;
- CLIP target probability at least `0.60`;
- CLIP target-to-negative margin at least `0.45`.

The final live reacquisition retained `13/18` monitor frames. Replay retained
`18/23` frames from the preceding positive capture and rejected `0/17` frames
from the clear wide-wall view. The ambiguous upper-left view was accepted in
`17/17` frames and must not be counted as a false-positive measurement until the
physical object is identified.

This is enough for continued camera-only and shadow testing, but it is not
evidence of general object grounding. Each new mission noun needs positive and
physically verified hard-negative calibration before it can control flight.

The corrected gray4 image remains build-verified only until it receives the same
moving hardware gate.

### Height-controlled monitor shadow flight

The `0.55 m` shadow flight produced no monitor detections because the camera saw
under the desk; only the bottom edge of the displays entered the top pixels. A
later `1.0 m` run produced `8/24` in-flight detections, but the desk height was
changed during the run and the result cannot be used as a controlled comparison.

With the desk fixed at its raised height, the final `1.3 m` run produced:

- `20/21` sampled in-flight frames with accepted monitor detections;
- accepted boxes aligned with the physical displays across the `45` degree yaw
  sweep and return;
- altitude between `1.301` and `1.378 m` during the logged sequence;
- maximum estimated horizontal drift of `0.233 m`;
- battery sag to `3.669 V`, with no tumbled flag.

The post-flight check reported motors zero, no flying or tumbled state, and
`20%` battery. Charge before any further flight.

### First recurrent Puffer semantic checkpoint

The first accepted semantic-navigation checkpoint is:

```text
artifacts/puffer_semantic/semantic_nav_seed726_8192.pt
SHA-256 727d7e02b54247c9e1cc38b68a2f7efb39e61ce51ec1bdc01300debe8f3b4c1e
```

It was trained from randomized MuJoCo rooms and simulator expert actions, not
from supervised labels copied from the earlier real flights. The actor has
59,745 parameters and consumes `64x48` gray4 appearance/delta/motion, a
`16x16` egocentric semantic map at `0.5 m/cell`, 13 proprioceptive values, and
a three-category command token. A CNN, target-evidence centroid, and MinGRU
produce firmware-level body velocity, vertical velocity, and yaw-rate
setpoints. Crazyflie firmware remains responsible for stabilization.

The PufferLib PPO continuation degraded the bootstrap policy and was rejected.
The selected simulator-bootstrap state achieved the following over 16 episodes
in four unseen rooms:

- full observation: `31.25%` success and `0%` collision;
- target evidence removed: `0%` success;
- raw vision removed: `18.75%` success;
- command token rotated: `31.25%` success.

The command result is expected at this stage: Grounding DINO and CLIP resolve
the text prompt into the target-evidence map before the actor. This is a
camera-grounded semantic-navigation policy, not yet an independently
language-grounded actor.

The checkpoint also replayed 86 frames from an earlier real QVGA flight,
including 13 accepted monitor detections. It produced finite proposals, but
the raw median proposal was about `0.26 m/s` forward and `20 deg/s` yaw, so it
is approved only for shadow logging. The reusable replay artifacts are:

```text
artifacts/puffer_semantic/semantic_nav_seed726_8192_live_replay.csv
artifacts/puffer_semantic/semantic_nav_seed726_8192_live_replay.summary.json
```

The next live run must use the corrected semantic QVGA profile because the
host grounder needs more than the optimized `64x48` gray4 stream. The policy
adapter downsamples QVGA back to its exact `64x48` contract:

```bash
AIDECK_UDP_FLASH_CONFIRM=FLASH_AIDECK_UDP \
  scripts/aideck_udp_streamer.sh flash-semantic-highres

uv run --extra semantic --extra hardware \
  python scripts/crazyflie_semantic_find.py \
  --prompt "computer monitor" \
  --shadow-checkpoint artifacts/puffer_semantic/semantic_nav_seed726_8192.pt \
  --height-m 0.8 \
  --duration-s 20 \
  --flight \
  --confirm-flight \
  --confirm-semantic-yaw-control
```

This command leaves bounded reposition disabled. The existing discovery
controller controls only yaw while firmware holds height; every Puffer proposal
is written under `policy_shadow` with `controls_drone=false`. Recheck battery,
deck detection, frame integrity, and monitor visibility at or below `0.8 m`
immediately before launching.

### Yaw-only Puffer live gate

The visibility-conditioned v2 checkpoint reached the next live-flight gate:

```text
artifacts/puffer_semantic/semantic_nav_v2_seed726_8192.pt
SHA-256 65fce7eea317e52e32f13ce8b421bd73a9f54895c325640117478ca1562705fa
```

It passed the recorded-flight replay gate with zero pre-acquisition and
detection-suppressed translation, `100%` directional yaw agreement, detected
yaw capped at `8 deg/s`, and search yaw below `20 deg/s`. The unseen-room
simulation gate had zero collisions and correct bounded yaw, but zero complete
missions. Therefore this checkpoint is approved only for yaw authority.
Firmware stabilization, takeoff, altitude hold, timeout, and landing remain
unchanged, while learned `vx`, `vy`, and `vz` are hard-clamped to zero.

The generated gate contract is:

```text
artifacts/puffer_semantic/semantic_nav_v2_seed726_8192_next_live_readiness.json
```

It binds the evidence to the checkpoint hash. The live command refuses a
mismatched checkpoint, a failed report, translation authority, missing
confirmation, or bounded exploration. The next live run, after the user
confirms the drone is charged and positioned, is:

```bash
uv run --extra semantic --extra hardware \
  python scripts/crazyflie_semantic_find.py \
  --prompt "computer monitor" \
  --shadow-checkpoint \
    artifacts/puffer_semantic/semantic_nav_v2_seed726_8192.pt \
  --puffer-readiness-report \
    artifacts/puffer_semantic/semantic_nav_v2_seed726_8192_next_live_readiness.json \
  --puffer-yaw-control \
  --height-m 0.8 \
  --duration-s 20 \
  --flight \
  --confirm-flight \
  --confirm-semantic-yaw-control \
  --confirm-puffer-yaw-control
```

This is the first policy-authority gate, not a navigation success claim.
Forward mission control stays disabled until an unseen-room simulator gate
demonstrates reliable mission completion without collisions.

#### First yaw-authority flight

The gated run completed on 2026-07-27:

```text
artifacts/semantic/20260727T152639Z-computer-monitor
```

The drone flew for `20 s` at a `0.8 m` target height with no safety abort.
Puffer yaw authority was active for all 57 processed frames, while every
applied and logged translation command remained exactly zero. The host
grounder accepted a monitor in 51 frames. Selected annotated frames show boxes
on the physical displays rather than the bright floor or window region.

The policy spent only six frames in no-detection search because a monitor
entered view early. It reached the `20 deg/s` search cap and the `8 deg/s`
detected-target cap, then settled to a median yaw command of `0.28 deg/s`.
Telemetry estimated yaw from `-1.1` to `52.1` degrees, with a `-1.3` to `59.0`
degree range. This validates acquisition and tracking, not a complete
360-degree search.

Altitude remained between `0.762` and `0.919 m`; maximum relative horizontal
estimator displacement was `0.344 m`. Despite sunlight and reported airflow,
maximum absolute roll and pitch were `1.95` and `1.30` degrees. A separate
postflight sample recorded motors zero, `sys.isFlying=0`, `sys.isTumbled=0`,
and approximately `4.05 V`.

The next policy must not be trained or evaluated against a forced 360-degree
scan. Rotation is only one possible information-gathering action. The intended
mission is active semantic exploration: use recurrent map state to remember
observed and unexplored space, move through visually free space, avoid
obstacles, acquire the requested object whenever it enters view, and approach
it safely. The live orchestrator therefore defaults to no mandatory initial
scan and may complete as soon as a valid target is centered.

Room-scale translation remains disabled at this gate. It should receive live
authority only after the active-exploration policy demonstrates randomized-room
mission completion, collision avoidance, target-hidden recovery, and bounded
setpoint behavior in simulation and recorded-flight replay.
