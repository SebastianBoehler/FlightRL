# Crazyflie hardware bring-up

This guide covers explicit-profile inspection, telemetry, a nonlearned scripted
demo, and propeller-off motor diagnostics. There is no learned-policy launcher
in the reviewed tree. The edge-v3 PyTorch actor is a Mac reference, not a GAP8
artifact or flight checkpoint.

## Install and choose the exact profile

```bash
python -m pip install -e ".[hardware,dev]" --no-build-isolation
```

Every hardware command requires `--config`. Select the file matching the
physical aircraft and deck stack; never infer deck presence from a previous
run or filename.

- `configs/hardware/crazyflie_2_1_brushless_flow_only.toml`
- `configs/hardware/crazyflie_2_1_brushless_aideck_flow2.toml`
- `configs/hardware/crazyflie_2_1_brushless.toml` for the separately defined
  base stack

The AI Deck profile explicitly expects AI Deck, Flow Deck v2, and Z-ranger and
does not expect a Multi-ranger.

## Propeller-off checks

Charge and inspect the battery, remove propellers, mount only the decks named by
the selected profile, and place the aircraft still on a clear surface.

Dry-run validates parsing/control flow without importing `cflib`:

```bash
python scripts/crazyflie_bringup.py \
  --config configs/hardware/crazyflie_2_1_brushless_aideck_flow2.toml \
  --dry-run check
```

Read-only connection and deck/supervisor inspection:

```bash
python scripts/crazyflie_bringup.py \
  --config configs/hardware/crazyflie_2_1_brushless_aideck_flow2.toml \
  check
```

Stop on any unexpected/missing deck, estimator/supervisor failure, low battery,
link instability, timeout, or nonfinite telemetry. Resolving a historical URI
or connecting once does not clear the aircraft.

## Telemetry

Validate then record with the same explicit profile:

```bash
python scripts/crazyflie_log.py \
  --config configs/hardware/crazyflie_2_1_brushless_aideck_flow2.toml \
  --duration-s 10 \
  --dry-run

python scripts/crazyflie_log.py \
  --config configs/hardware/crazyflie_2_1_brushless_aideck_flow2.toml \
  --duration-s 10 \
  --output artifacts/crazyflie_logs/bringup.csv \
  --console-output artifacts/crazyflie_logs/bringup_console.jsonl
```

The logger filters requested variables against the connected firmware TOC but
fails if no usable log blocks remain. Record the firmware/deck/config identity
with data used for calibration or replay; an available variable name does not
guarantee identical units or semantics across firmware revisions.

## Propeller-off motor bench

This is a material hardware operation even without propellers. First review the
dry-run plan:

```bash
python scripts/crazyflie_motor_bench.py \
  --config configs/hardware/crazyflie_2_1_brushless_flow_only.toml \
  --dry-run
```

A real run additionally requires `--confirm-props-off`, supervisor approval,
bounded powers, live telemetry, and the watchdog. It zeros all motors and
disarms in cleanup. Execute it only when the physical setup and intended output
are separately confirmed.

## Scripted demo boundary

`crazyflie_bringup.py ... demo --confirm` uses the firmware stabilizer for a
short takeoff/hover/turn/land sequence from the selected config. It is not a
policy test. Props-on execution requires a fresh physical checklist, protected
area, charged battery, operator abort, and explicit approval for that run.

Do not proceed from a passing `check`, telemetry log, simulator gate, teacher
result, dry-run, or desktop export to physical motion automatically.

## Learned-policy boundary

Learned proposals remain blocked until the exact edge-v3 student, float/int8/
GAP8 parity, measured target budget, CPX sequence/freshness protocol, STM32
safety consumer, typed deployment bundle, and staged hardware evidence all
exist and pass. Generic manifests and legacy checkpoint names cannot authorize
control.
