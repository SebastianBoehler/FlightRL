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
