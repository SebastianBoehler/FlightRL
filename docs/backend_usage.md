# FlightRL Backend Usage

Use the Python, native, and Puffer paths for different jobs.

## Python 6-DoF Env

Use for correctness work:

- debugging dynamics, observations, rewards, and resets
- writing parity tests
- prototyping new tasks before moving them native
- checking behavior with small `num_envs`

The Python env is readable and flexible, but it is not the target for large training runs.

## Native 6-DoF Env

Use for main simulation training:

- PPO and DAgger rollout collection
- high-throughput benchmark sweeps
- task/reward experiments once parity tests exist
- sim-only checkpoint generation

The native env is the fast execution backend. Keep the Python env as the reference and require parity tests when changing native behavior.

## Puffer Export

Use for PufferLib integration work:

- exporting FlightRL as a Puffer/Ocean-style environment
- benchmarking against Puffer-native environments
- preparing an upstream/demo contribution path

The Puffer export should reuse the same native 6-DoF core instead of duplicating reset, reward, or observation logic.

## Hardware Scripts

Use only for supervised Crazyflie work:

- deck checks and telemetry logging
- motor bench and calibration flights
- replay evidence collection
- manually confirmed scripted demos

Learned checkpoint control must stay behind sim-to-real approval manifests and hardware blockers. Do not use hardware scripts for unattended experiments.

## Rule Of Thumb

- Need to understand or change behavior: start in Python.
- Need to train fast: use native.
- Need to package for Puffer: use Puffer export.
- Need to touch the real drone: use hardware scripts only after safety gates pass.
