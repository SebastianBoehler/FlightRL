# Native visual training fast lane

Status: simulation and non-actuating live shadow measured 2026-07-27. No
checkpoint in this document is approved for live flight authority.

## First-principles boundary

The training hot loop is:

1. native six-DoF physics,
2. native procedural room and obstacle sampling,
3. native low-resolution ray rendering,
4. native temporal observation assembly,
5. native reward, terminal, and reset logic,
6. PufferLib vector collection and PPO,
7. one small PyTorch encoder plus MinGRU policy.

Python configures a run, loads PufferLib, evaluates checkpoints, and writes the
report. It is not used to step individual environments or render frames.
MuJoCo remains useful for slower high-fidelity validation, not bulk training.

The policy controls body-frame velocity, vertical velocity, and yaw-rate
residuals. Crazyflie firmware remains responsible for stabilization.

## Measured bottleneck

At `16x12`, 256 environments, and eight CPU worker threads on the M4 Max:

| Path | Measured throughput |
| --- | ---: |
| Native environment including rendering | 2,050,595 transitions/s |
| Previous end-to-end PPO path | about 2,200 transitions/s |
| Maximum-throughput PPO profile | 220,000-292,000 transitions/s |
| Learning-oriented PPO profile | about 81,000 transitions/s |
| Learning profile rollout component | about 500,000-580,000 transitions/s |
| Learning profile optimizer component | about 95,000 transitions/s |

The renderer was not the original bottleneck. The large loss came from running a
convolutional encoder on 192 pixels, collecting only 16 recurrent steps at a
time, and paying optimizer overhead on small batches.

The maximum-throughput profile uses one optimizer pass over each 16,384-sample
batch. It is useful for throughput measurement but did not give the policy
enough gradient updates. The learning profile uses:

- 256 environments
- 64-step recurrent rollouts
- 4,096-sample minibatches
- replay ratio 4
- 32-wide MinGRU state
- flat 192-to-32 visual projection at `16x12`
- learned action log standard deviation clamped to `[-2, 0]`

Higher resolutions retain the convolutional encoder. Resolution is a build-time
property of the C environment, so the native renderer and observation buffers do
not carry a dynamic-shape branch in their inner loops.

## Superseded causal result

The scene generator alternates a wall-adjacent barrier with a random open side.
A fixed left or right dodge therefore cannot solve every episode. Texture and
sensor noise stay fixed within an episode; only pose changes generate temporal
image differences.

Checkpoint:

`artifacts/puffer_visual/flightrl_visual_fast16_bounded_8388608.bin`

Report:

`artifacts/puffer_visual/flightrl_visual_fast16_bounded_8388608.report.json`

Training used 8,388,608 transitions in 103.5 seconds. Deterministic evaluation:

| Evaluation | Success | Collision |
| --- | ---: | ---: |
| Obstacle, full camera | 45.32% | 54.52% |
| Obstacle, camera masked | 0.00% | 53.61% |
| Clear room, full camera | 73.64% | 0.00% |

This is evidence that the action depends causally on pixels. It is not sufficient
for a live flight gate. The camera-masked failure is desirable evidence; the
full-camera collision rate is not.

## Randomized low-light candidate

The current candidate adds episode-level room dimensions, wall-side obstacle
layout, mass, drag, rate response, thrust, motor response, exposure, material,
texture, and horizontal-lighting variation. Training uses 75% obstacle rooms
and 25% clear rooms. Image noise is static within an episode, so it does not
create false motion.

The native simulator writes one privileged expert label after the policy
observation. `FlightRLVisionEncoder` excludes that value from its visual and
intent slices. The exported runtime omits it entirely. A 32-step parity test
between PufferLib's 583-value training path and the standalone 582-value
deployment path produced zero action, value, and recurrent-state error.

Checkpoint:

`artifacts/puffer_visual/flightrl_visual_fast16_lowlight_v5_1048576.bin`

SHA-256:

`e599a351901d65336a08484457f39de9250c6f756573a433dd6308eb6eed27ed`

The selected 33,761-parameter bootstrap policy used 1,048,576 PPO transitions
and 384 recurrent bootstrap updates. Held-out evaluation:

| Evaluation | Success | Collision | Lateral action p95 |
| --- | ---: | ---: | ---: |
| Randomized obstacle, full camera | 98.59% | 1.33% | 1.000 |
| Randomized obstacle, camera masked | 0.00% | 100.00% | 0.213 |
| Randomized clear room | 99.84% | 0.00% | 0.320 |
| Nominal obstacle | 98.51% | 1.41% | 1.000 |

The observation path now performs gray4 quantization followed by per-frame
contrast normalization. The simulated exposure range includes mean intensity
18-110. This is software low-light support; zero-light operation still needs
illumination such as an IR source.

An initial nighttime shadow exposed camera-profile drift: the deck supplied
`324x244` JPEG at 13.80 FPS. After restoring the corrected frame-safe gray4
image, the final non-actuating gate recorded:

- 650 `64x48` gray4 frames at 64.86 FPS with zero drops
- healthy low-light input at mean intensity 37.2
- 0.8% p95 temporal delta and 0.5% p95 motion-mask activity
- 0.217 ms median and 12.06 ms maximum Mac inference
- zero non-lateral p95 action
- bounded lateral p95 `0.0056 m/s` and maximum `0.0222 m/s`

All stationary live-shadow gates pass. The artifact is
`artifacts/puffer_visual/flightrl_visual_fast16_lowlight_v5_policy_profile_gate.summary.json`.
This evidence alone grants no flight authority.

## Bounded live waypoint gate

`scripts/crazyflie_visual_waypoint_control.py` prepares the first actuating
gate without replacing the Crazyflie flight stack:

- firmware performs arming, takeoff, attitude stabilization, altitude hold,
  setpoint execution, and landing
- a deterministic body-frame controller tracks one 0.30 m forward waypoint
- the learned checkpoint controls only the lateral body-velocity residual
- the first-run policy blend limits that residual to 0.0192 m/s
- vertical and yaw policy outputs are logged but ignored
- camera age, low-light quality, Flow Deck presence, telemetry, attitude,
  height, cross-track, displacement, battery, and checkpoint hashes are gated

The runner requires both the simulation report and stationary live-shadow
report to match the checkpoint. Its dry run is non-actuating:

```bash
PYTHONPATH=. uv run python scripts/crazyflie_visual_waypoint_control.py --dry-run
```

The live command additionally requires explicit flight, policy-authority,
clear-path, and room-light confirmations. It must only be run after the drone
is placed on the floor and the operator confirms readiness. This gate tests
local visual waypoint correction; it does not perform semantic object search.

## Run

The learning-oriented values are now the script defaults:

```bash
uv run --with rich-argparse --with pybind11 \
  python scripts/train_puffer_visual_navigation.py \
  --env-name flightrl_visual_fast16_next \
  --obstacle-probability 1 \
  --reset-action-head
```

Each report includes end-to-end SPS, rollout SPS, update SPS, optimizer sample
SPS, full-camera evaluation, camera-masked evaluation, and clear-room
evaluation.

Run a non-actuating live camera shadow with:

```bash
uv run python scripts/crazyflie_visual_puffer_shadow.py \
  --checkpoint artifacts/puffer_visual/flightrl_visual_fast16_lowlight_v5_1048576.bin \
  --output artifacts/puffer_visual/flightrl_visual_fast16_lowlight_v5_live_shadow.csv
```

## External references

[`tensaur/drone`](https://github.com/tensaur/drone) is useful for compact drone
dynamics, Crazyflie firmware integration, and a generated C PufferNet forward
pass. Its current training observations are state values, not camera pixels.

[`Emerge-Lab/DroneRacing`](https://github.com/Emerge-Lab/DroneRacing) adds
native racing and swarm environments, but its policy observations are also
privileged state. Raylib is used for human visualization rather than
policy-camera rendering.

FlightRL should reuse their low-level dynamics and firmware ideas, not add them
as another simulator abstraction around the existing native environment.

## Next gates

1. Run the 0.30 m bounded lateral-authority waypoint gate and inspect the
   synchronized camera, odometry, command, and abort data.
2. Add randomized native scene primitives and semantic target IDs to the same C
   renderer. Language should be converted outside the control loop to a compact
   target condition.
3. Add camera-pose, latency, and dropped-frame randomization without per-frame
   scene flicker.
4. Export the compact recurrent policy to a C forward pass and benchmark it
   separately from capture and control.
5. Move physics, rendering, and inference to a fused GPU path only if more than
   the current approximately 80,000 learning transitions/s is materially useful.
