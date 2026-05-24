# PULP-DroNet 2019 Paper Analysis

Paper: [An Open Source and Open Hardware Deep Learning-powered Visual Navigation Engine for Autonomous Nano-UAVs](https://arxiv.org/abs/1905.04166)

Related project: [pulp-platform/pulp-dronet](https://github.com/pulp-platform/pulp-dronet)

## Executive Summary

This paper is highly relevant to FlightRL's sim-to-real direction, but it should not be treated as a direct implementation target yet. The strongest fit is architectural: the paper validates a hierarchical split where a lightweight onboard perception model emits slow navigation commands, while the drone's existing flight stack handles stabilization and motor control.

For FlightRL, the main takeaway is to add an obstacle-aware command policy path before attempting end-to-end motor control from images. The paper's DroNet-style output contract is especially useful: predict collision probability and steering, then convert those outputs into bounded velocity and yaw-rate setpoints with simple filtering and safety limits.

The current repository already points in this direction through the Crazyflie hardware layer, range-based avoidance policy, and sim-to-real roadmap. The missing pieces are a real obstacle task in simulation, 6-DoF dynamics, richer sensor models, replay comparison against Crazyflie logs, and eventually a camera or AI-deck perception path.

## What The Paper Actually Shows

The authors deploy a visual navigation stack on a 27 g Crazyflie-class nano-UAV using a GAP8-based PULP-Shield. Their model is based on DroNet, a compact residual CNN trained to map monocular grayscale camera frames to:

- collision probability
- desired steering direction

Those model outputs are not sent directly to motors. They are translated into target forward velocity and target yaw rate with low-pass filtering:

- high collision probability reduces forward velocity
- steering output changes yaw-rate target
- the existing Crazyflie firmware remains responsible for low-level control

The reported field results are strong for the size and power class:

- onboard CNN inference at 6-18 fps
- 64-272 mW compute power depending on operating point
- autonomous traversal of a previously unseen 113 m indoor path
- dynamic obstacle avoidance at about 1.5 m/s when the obstacle appears 2 m ahead

The paper is as much about deployability as learning quality. Important engineering steps include dataset fine-tuning with the actual onboard camera, fixed-point quantization, batch-normalization folding, and minimizing communication between the flight MCU and perception accelerator.

## Fit With FlightRL

### Strong Fit

FlightRL's intended autonomy split already matches the paper:

```text
camera / range / telemetry -> semantic navigator -> command policy -> stabilizer -> motor mixer
```

The paper reinforces that this is the right boundary. Perception should produce compact navigation intent, not raw motor commands.

The Crazyflie target also matches the repository. `docs/crazyflie_bringup.md` already focuses on safe setpoint-level control with Flow deck and Multi-ranger telemetry, and `src/flightrl/hardware/avoidance_policy.py` already has an obstacle-avoidance policy that emits velocity, yaw-rate, and height commands from range readings.

The paper's deployment discipline is also a good match: no external positioning dependency, no remote compute dependency for the final loop, and clear attention to latency, power, and onboard execution.

### Partial Fit

The current simulator is planar. That is still useful for fast training experiments, but it cannot faithfully evaluate the paper's closed-loop behavior because the paper depends on 3D yaw/velocity/altitude setpoints and physical Crazyflie dynamics.

The current observation schema has a `include_vision_sensor` flag, but `VISION_SENSOR_DIM = 0` and config loading raises `NotImplementedError` for vision. That is a good fail-fast posture, but it means DroNet-style image inputs are future work.

The current range-based avoidance path is closer to an immediate implementation target than full image-based PULP-DroNet. We can first reproduce the paper's output contract using simulated or real Multi-ranger observations, then later swap the input encoder from range features to camera features.

### Weak Fit

The GAP8/PULP-Shield optimization details are not immediately useful unless FlightRL targets Bitcraze AI-deck or another embedded inference board. For now, the quantization and batch-norm folding sections are better treated as future deployment requirements, not training requirements.

The paper's original training data comes from driving and bicycle datasets, then gets fine-tuned with onboard camera data. That is not a clean fit for FlightRL's current simulator because the repository has no camera renderer, no visual domain randomization, and no labeled image dataset pipeline.

## What We Should Implement

### 1. Add A DroNet-Style Command Interface

Add a small interface for perception policies that emit:

- `collision_probability`
- `steering`
- optional confidence or validity flag

Then convert this into bounded Crazyflie-compatible setpoints:

- forward velocity
- lateral velocity if using range/deck data
- yaw rate
- target height or vertical velocity

This can reuse the existing safety shape from `AvoidanceCommand`. The key improvement is making the output contract explicit and independent of whether the upstream model uses ranger telemetry, synthetic observations, or camera frames.

### 2. Add A Sim Obstacle-Avoidance Task

FlightRL needs an obstacle task before vision work is useful. Start with simple geometry:

- corridor walls
- static box obstacles
- one dynamic obstacle that appears within a configurable distance
- success measured by distance traveled without collision

This would let us reproduce the spirit of the paper's control evaluation without pretending to have a camera model.

### 3. Add Collision Probability As A Learned Auxiliary Head

The paper's two-head structure is useful even if we train with RL. FlightRL could train a policy with:

- action output for control
- value output for RL
- auxiliary collision-risk prediction head

The auxiliary target can come from simulator geometry or near-future collision rollout. This gives the learned controller a safety-relevant representation without requiring full visual learning.

### 4. Add Setpoint-Level Action Mode

The paper does not send direct motor commands from the CNN. FlightRL should add a command-level action mode for:

- target forward velocity
- target lateral velocity
- target yaw rate
- target altitude or vertical velocity

This aligns sim training with the safe hardware boundary already described in the bring-up docs.

### 5. Add Real-Log Replay And Scoring

The paper evaluates closed-loop behavior with onboard logs and physical tests. FlightRL should add replay tooling that compares Crazyflie logs against simulated rollouts:

- range readings
- estimated position or flow velocity
- commanded setpoints
- battery voltage
- collision/clearance margins

This should become the gate before any learned policy is allowed near real hardware.

### 6. Defer Camera-Based DroNet Until The Control Stack Is Ready

Vision should be a second-stage effort. Before implementing image observations, FlightRL should have:

- 6-DoF dynamics
- setpoint action mode
- obstacle task
- range sensor simulation
- replay comparison
- hardware safety gates

After that, a camera path can be added using either a lightweight synthetic renderer or real AI-deck / camera dataset collection.

## Candidate GitHub Issues

### Issue 1: Add Obstacle-Avoidance Task To Native Simulator

Implement a native C task for corridor and obstacle avoidance. Include static obstacles, simple dynamic obstacle spawn, collision termination, clearance reward, and distance-traveled success metrics.

Acceptance criteria:

- new TOML config under `configs/tasks/`
- task enum and native task/reward/termination support
- tests for collision, success, and reward direction
- no vision mock data

### Issue 2: Add Command-Level Setpoint Action Mode

Add an action mode that represents velocity/yaw/altitude setpoints instead of direct planar thrust or motor commands.

Acceptance criteria:

- config action mode such as `velocity_yaw_height`
- bounded action conversion in native and Python wrappers
- documentation explaining hardware-safe setpoint boundary
- tests for action dimension, clipping, and export config

### Issue 3: Introduce Perception Command Output Types

Create a small policy interface for obstacle/navigation heads that output collision probability and steering, then map them into bounded setpoints.

Acceptance criteria:

- dataclass for collision/steering output
- converter to `AvoidanceCommand`-like setpoints
- low-pass filtering parameters
- unit tests for collision braking and steering/yaw mapping

### Issue 4: Train Collision-Risk Auxiliary Head

Add optional auxiliary collision-risk prediction during policy training or imitation training.

Acceptance criteria:

- simulator-provided collision-risk target
- model head or separate small module
- logged auxiliary loss
- ablation config that can disable the head

### Issue 5: Add Crazyflie Log Replay Evaluation

Add a script that loads real Crazyflie CSV logs and scores a simulated or learned policy against recorded sensor states.

Acceptance criteria:

- reads existing `artifacts/crazyflie_logs/*.csv`
- reports command error, clearance margin, saturation, and emergency-stop indicators
- exits nonzero for unsafe thresholds
- no generated fake logs

### Issue 6: Research AI-Deck / PULP-DroNet Integration Path

Document whether FlightRL should target Bitcraze AI-deck, host-side camera inference, or both.

Acceptance criteria:

- compare AI-deck, host laptop, and pure range-deck paths
- identify deployable model format and latency budget
- identify required data collection format
- recommend first implementation path

## Recommended Sequence

1. Implement obstacle avoidance with range-style observations.
2. Add setpoint-level action mode.
3. Add collision/steering output types and low-pass command mapping.
4. Add replay evaluation from real Crazyflie logs.
5. Train and test range-based policies in simulation.
6. Only then start camera or AI-deck work.

This sequence borrows the paper's core control architecture while respecting FlightRL's current maturity. It avoids jumping straight into embedded vision before the simulator, setpoint interface, and hardware safety gates can evaluate the resulting behavior.

## Key Risks

- A planar environment may teach avoidance behavior that fails in real 3D yaw/roll/pitch dynamics.
- A model trained only on synthetic obstacle geometry may overfit to simplified range readings.
- Vision training without real camera data is likely to look good in simulation and fail on hardware.
- Direct motor policies remain a poor hardware deployment target until the simulator has 6-DoF dynamics and validated actuator models.
- Embedded inference optimization is premature unless the target deployment board is selected.

## Bottom Line

The paper is a strong strategic fit. It supports FlightRL's current direction toward hierarchical autonomy and safe setpoint-level hardware control. The most useful near-term change is not to port PULP-DroNet directly, but to implement the same output contract and evaluation mindset: obstacle-aware perception produces bounded command setpoints, the stabilizer handles fast control, and real-log replay gates hardware deployment.
