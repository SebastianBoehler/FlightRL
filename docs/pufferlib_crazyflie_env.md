# PufferLib Crazyflie 6-DoF Env

## Goal

Prepare FlightRL's native Crazyflie-style 6-DoF environment as a small upstream-quality PufferLib Ocean contribution.

The intended contribution is a compact high-throughput RL benchmark environment, not a complete hardware simulator. MuJoCo, Newton, Isaac Lab, Flightmare, and PyBullet should remain validation or comparison lanes unless benchmarks show they should become part of the training path.

## Upstream Pitch

`crazyflie_6dof` is a native C/Ocean quadrotor control environment with a Crazyflie-like scale, 28-value observation, 4-value action, box-room range sensors, tunable physical parameters, and cheap per-env domain randomization.

It is useful for:

- PPO/SAC-style algorithm benchmarking on a continuous-control aerial task.
- High-throughput domain-randomization experiments.
- Testing sim-to-real-style policy robustness without requiring hardware, MuJoCo, or Isaac Sim.
- Providing a small open drone env for PufferLib developers.

## Current Contract

Observation shape: `28`

Action shape: `4`

Action meaning:

- `action[0]`: normalized collective thrust command.
- `action[1]`: normalized roll-rate command.
- `action[2]`: normalized pitch-rate command.
- `action[3]`: normalized yaw-rate command.

Native physics row:

- `mass_kg`
- `gravity_m_s2`
- `linear_drag`
- `rate_tau_s`
- `thrust_scale`
- `max_rate_roll`
- `max_rate_pitch`
- `max_rate_yaw`
- `motor_tau_s`

## Immediate Checklist

- Keep the FlightRL Python reference env and native C env in parity.
- Keep all native env files small and reviewable.
- Keep the generated Ocean env dependency-free: no Crazyflie hardware, MuJoCo, Newton, Isaac, or PyBullet dependency.
- Verify export shape with `scripts/build_sixdof_puffer_export_report.py`.
- Verify throughput with `scripts/benchmark_sixdof_native.py`.
- Document known limitations before proposing an upstream PR.

## Verification Commands

```bash
python -m pytest -q tests/test_sixdof_physics.py tests/test_puffer4_export.py tests/test_sixdof_benchmark.py

python scripts/build_sixdof_puffer_export_report.py \
  --output artifacts/replay/sixdof_puffer_export_report.json

python scripts/benchmark_sixdof_native.py \
  --num-envs 1024 \
  --steps 1000 \
  --physics-profile crazyflie_brushless \
  --domain-randomization crazyflie_training
```

## Known Limitations

- The env is still a compact benchmark simulator, not a certified Crazyflie digital twin.
- Contacts are limited to room-bound termination/range geometry; no landing or collision response is modeled yet.
- Sensor modeling is still simple and should be expanded from real logs.
- Domain-randomization ranges are plausible starting points, not fitted distributions.
- Onboard deployment requires a separate policy-size, latency, and safety gate.

## Later Work

- Compare the native env against MuJoCo rollouts and real Crazyflie logs.
- Treat MuJoCo XML, solver, actuator, and sensor settings as sweepable hyperparameters.
- Spike Newton/Warp only after MuJoCo parity and replay-error tooling are useful.
- Use Flightmare or Isaac Sim when camera/perception/VLA data becomes the main bottleneck.
- Add an upstream PufferLib branch once the export report, benchmark table, and limitations are stable.
