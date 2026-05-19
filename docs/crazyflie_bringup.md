# Crazyflie Hardware Bring-Up

This guide prepares FlightRL for a Bitcraze Crazyflie 2.1 Brushless with Flow deck v2 and Multi-ranger deck. The current hardware layer is for safe scripted bring-up and telemetry collection only. Learned policies must stay in simulation until FlightRL has a 6-DoF model, replay comparison, and hardware safety gates.

## Install

```bash
python -m pip install -e ".[hardware,dev]" --no-build-isolation
```

The `hardware` extra installs Bitcraze `cflib`. It is optional because normal simulator training and tests do not need radio hardware.

## Physical Checklist

- Update and test the Crazyflie with the official Bitcraze client before using FlightRL scripts.
- Put the Flow deck v2 underneath the drone.
- Put the Multi-ranger deck above the drone.
- Run all checks with propellers off first.
- Use a clear indoor area with no people close to the flight path.
- Keep the drone within reach for power removal and be ready to interrupt the script.

## Config

The default config is:

```text
configs/hardware/crazyflie_2_1_brushless.toml
```

It uses the default Bitcraze radio URI `radio://0/80/2M/E7E7E7E7E7`, a 0.3 m hover height, conservative velocity, Flow deck v2 expectation, Multi-ranger expectation, and `requires_manual_confirm = true`.

## Bring-Up Commands

Dry-run commands do not import `cflib`, do not scan radio hardware, and do not write fake telemetry.

```bash
python scripts/crazyflie_bringup.py --dry-run scan
python scripts/crazyflie_bringup.py --dry-run check
python scripts/crazyflie_bringup.py --dry-run demo
```

When the replacement board is ready, scan and check the drone:

```bash
python scripts/crazyflie_bringup.py scan
python scripts/crazyflie_bringup.py check
```

Only run the demo with props on after the prop-off checks pass:

```bash
python scripts/crazyflie_bringup.py demo --confirm
```

The demo takes off to roughly 0.3 m, hovers, turns left and right in place, hovers again, and lands.

## Telemetry

Dry-run logging validates config and prints the intended output path without creating fake rows:

```bash
python scripts/crazyflie_log.py --dry-run
```

Real logging writes replay-friendly CSV under `artifacts/crazyflie_logs/`:

```bash
python scripts/crazyflie_log.py --duration-s 10
```

CSV columns start with `host_time_s` and `crazyflie_time_ms`, followed by configured cflib log variables. These logs are intended for later parameter fitting and sim-to-real replay checks.

## Ranger Hold Policy

The current learned hardware policy runs above the Crazyflie firmware stabilizer. It emits bounded velocity, vertical velocity, and yaw-rate setpoints; it does not command motors directly.

Train the checkpoint:

```bash
python scripts/train_ranger_hold.py --checkpoint artifacts/checkpoints/ranger_hold.pt
```

Dry-run the loader and command serialization:

```bash
python scripts/crazyflie_hold_policy.py \
  --checkpoint artifacts/checkpoints/ranger_hold.pt \
  --dry-run
```

After charging the battery and checking that the Bitcraze client is closed, run a cautious live test:

```bash
python scripts/crazyflie_hold_policy.py \
  --checkpoint artifacts/checkpoints/ranger_hold.pt \
  --confirm-flight \
  --duration-s 15 \
  --max-speed-m-s 0.20 \
  --max-vertical-speed-m-s 0.14
```

By default, the live runner captures the current `stateEstimate.x/y` after takeoff and holds that point. Pass `--target X Y Z` only when you intentionally want it to fly toward an explicit world-frame target.

## RL Boundary

The current simulator is still planar and useful for fast hover/reach experiments. Hardware policy deployment must stay at the setpoint layer until FlightRL has a 6-DoF model, replay comparison, and broader safety gates. Learned policies should emit bounded velocity, altitude, and yaw setpoints through cflib; they should not command direct motors.
