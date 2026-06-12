# FlightRL Sim-To-Real Roadmap

## Goal

Build an indoor-first drone learning stack that can train fast stabilization and command-following policies in simulation, validate them on a small real drone, and later scale the same architecture toward larger PX4-class or industrial platforms.

The target autonomy split is hierarchical:

```text
camera / range / telemetry -> semantic navigator -> command policy -> stabilizer -> motor mixer
```

The vision-language-action layer should issue slow, high-level commands such as "fly to the right corner", "hold position", "approach", "circle", or "land". The low-level stabilizer should stay small, fast, local, and safety-gated.

## Hardware Path

Start with a small indoor platform instead of a large drone.

Recommended first platform:

- Bitcraze Crazyflie 2.1+ with Flow deck v2 and Crazyradio 2.0.
- The current Bitcraze STEM Drone Bundle is about USD 320 and includes those parts.
- The STEM Ranging Bundle is about USD 410 and adds Multi-ranger obstacle sensing.

Budget note:

- A USD 100 class drone is attractive for camera experiments, but usually lacks the telemetry, low-level control access, repeatability, and safety hooks needed for serious sim-to-real RL.
- Tello-style drones can be useful for simple Python command or camera demos, but they are a weak fit for training and deploying custom stabilization policies.
- Crazyflie costs more, but it is the better research platform because it has an open ecosystem, logging, Python APIs, optical flow, range decks, firmware access, and a realistic indoor safety profile.

## Phases

### Phase 1: Make FlightRL Physically Useful

- Replace the planar model with full 6-DoF quadrotor state.
- Model roll, pitch, yaw, angular velocity, motor layout, yaw torque, thrust, drag, and gravity.
- Keep the existing planar configs as smoke-test tasks, but move real-hardware work to 6-DoF configs.
- Add command-level action modes before relying on raw motor actions.

### Phase 2: Add Real Actuator And Sensor Models

- Add motor RPM or normalized motor state with first-order lag.
- Fit thrust and torque curves from bench or manufacturer data.
- Add battery voltage scaling and actuator saturation.
- Add IMU bias/noise, optical-flow noise, range noise, latency, and packet loss.
- Make domain randomization operate on measured hardware parameters.
- Use the optional MuJoCo backend as a physics-reference lane for rigid-body/contact checks, then port only validated fast-path effects into the native C/Ocean env for sweep-scale training.

### Phase 3: Train Stabilization And Command Following

- Train hover, attitude hold, altitude hold, velocity tracking, yaw tracking, and local waypoint tasks.
- Prefer high-rate stabilizer policies over direct VLA-to-motor control.
- Use RL where it can react quickly to disturbances, but keep emergency limits and termination checks deterministic.
- Track success, crash rate, energy use, action smoothness, and recovery after perturbations.

### Phase 4: Build The Real-Hardware Bridge

- Implement a Crazyflie bridge for telemetry, command setpoints, logging, and emergency stop.
- Start with high-level setpoints instead of direct motor commands.
- Record every flight as replay data.
- Add parameter-fitting scripts that compare real logs to simulated rollouts.
- Gate deployment with a short preflight checklist and a maximum-risk profile.

### Phase 5: Add Perception And VLA Control

- Start with explicit room targets and range/flow telemetry.
- Add camera perception only after stable command following works.
- Convert language commands into structured goals, not motor actions.
- Use the VLA layer for semantic intent and the command policy for local execution.

Example:

```text
"fly to the right corner" -> target region -> local waypoint -> velocity/yaw/altitude setpoints -> stabilizer
```

### Phase 6: Scale Up

- Transfer the architecture, not the exact policy, to larger drones.
- Move from Crazyflie to PX4/ArduPilot-class hardware after indoor validation.
- Refit mass, inertia, actuator, drag, and sensor models for each platform.
- Keep the same evaluation harness and command API so larger drones become a parameter and integration problem, not a rewrite.

## Near-Term Issues

- Use `docs/crazyflie_bringup.md` for the first Crazyflie 2.1 Brushless setup, scripted MotionCommander demo, and telemetry logging.
- Implement 6-DoF quadrotor dynamics.
- Add motor/prop actuator model and parameter schema.
- Add Crazyflie hardware bridge research and bring-up checklist.
- Add command-level action mode for velocity, yaw, altitude, and local waypoint tracking.
- Extend replay calibration beyond log-quality gates and per-signal scale/bias toward physical parameter fitting.
- Add perception/VLA interface that emits structured goals.
