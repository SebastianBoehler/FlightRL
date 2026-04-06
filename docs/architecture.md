# Architecture

## Data Flow

1. Python loads a TOML config into typed dataclasses.
2. `DronePlanarEnv` converts that config into numeric keyword arguments for the native binding.
3. The C binding allocates per-env state, points into shared NumPy buffers, and stores sim/task/reward/sensor parameters.
4. `vec_step` advances every native environment, writes observations and rewards in place, and aggregates episodic metrics through `vec_log`.
5. `PuffeRL` consumes the wrapper directly without an extra Gymnasium emulation layer.

For the PufferLib 4 path, the flow forks after step 1:

1. FlightRL flattens the same TOML config into a PufferLib `4.0` `.ini`.
2. FlightRL copies the native simulator modules into a target `ocean/<env_name>/` directory in an upstream PufferLib checkout.
3. FlightRL generates a thin Puffer 4 `binding.c` adapter that wraps the existing single-agent simulator into the static `vecenv.h` interface.
4. Upstream `build.sh` compiles that exported environment into the PufferLib 4 backend.

## Native Boundaries

The native side is intentionally split into small modules:

- `native_actions.c`: normalized action handling and actuator smoothing
- `native_dynamics.c`: planar rigid-body integration
- `native_reset.c`: reset sampling and domain randomization
- `native_tasks.c`: hover, waypoint, and waypoint-sequence progress logic
- `native_reward.c`: transparent reward decomposition
- `native_termination.c`: crash, timeout, and bounds checks
- `native_observation.c`: configurable observation assembly
- `native_logging.c`: episodic metric aggregation
- `native_env.c`: `c_reset` and `c_step` orchestration

The binding layer is also modular:

- `binding_helpers.h`: NumPy/Python helpers
- `binding_env.h`: per-env handle lifecycle
- `binding_vec.h`: vectorized reset, step, log, and close
- `binding.c`: native config parsing and exported methods

The PufferLib 4 export path reuses the same simulator modules but swaps in a generated `binding.c` that targets `vecenv.h` instead of the legacy Python C extension entrypoints.

## C Core Vs Python Wrapper

The C core owns:

- world state
- drone state
- reward terms
- termination flags
- task progression
- reset sampling
- observation generation

The Python side owns:

- config loading
- action and observation space declaration
- extension handles
- training and evaluation scripts
- rollout export and plotting

This keeps training headless and fast while leaving experiment management readable on the Python side.

## Future Sim-To-Real Path

The key extension points for sim-to-real work are already separated:

- `drone` config for mass, inertia, thrust, drag, and limits
- `environment.action_mode` for switching between stabilized and actuator-centric control
- `sensors` for moving from ideal state to noisier derived channels
- `domain_randomization` for per-episode perturbations
- hardware profile TOMLs under `configs/hardware/`

The MVP does not implement real hardware IO. It only keeps the configuration and action/sensor abstractions ready for that later work.

## Future Hierarchical Autonomy Path

The intended autonomy split is hierarchical rather than end-to-end motor control:

- `camera + telemetry + mission context -> VLA navigator`
- `VLA navigator -> high-level command setpoints`
- `stabilizer/controller -> motor mixing -> quadrotor`

For this project, that means future perception-heavy models should target high-level commands such as:

- velocity targets
- heading or yaw targets
- altitude targets
- local waypoint updates
- discrete modes like `hold`, `approach`, `circle`, or `land`

The low-level stabilizer remains responsible for:

- attitude stabilization
- disturbance rejection
- actuator smoothing and limits
- emergency recovery behavior

This keeps the fast PufferLib/Ocean C backend useful for stabilization and control research while leaving room for a slower multimodal navigator on external compute later. It also avoids coupling future VLA work to unsafe or brittle direct motor-command outputs.

## Future Multi-Agent Path

The current MVP is single-drone, but the core already separates drone state from world/task state. To extend toward multiple drones later:

- store an array of `DroneState` structs instead of one drone in the env
- keep one shared world/task object per environment
- make task, reward, and sensor functions operate on `(env, drone_index)`
- expand the wrapper so `num_agents` becomes `num_envs * drones_per_env`

The current file layout is meant to make that refactor incremental instead of a rewrite.

## Deliberate MVP Omissions

- no live native renderer
- no obstacle avoidance task
- no range or vision simulation output
- no replay ingestion from real logs
- no multi-agent stepping

Unsupported features fail fast instead of silently substituting fake data.
