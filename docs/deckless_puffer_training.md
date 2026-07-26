# Deckless PufferLib Training

Use this lane while the Multi-ranger deck is unavailable. It keeps the same
6-DoF PufferLib policy/export path, but disables range observations explicitly
instead of substituting live ranger data.

## Training

Current passing deckless checkpoint:

```bash
python scripts/train_puffer_sixdof_bc.py \
  --checkpoint artifacts/checkpoints/puffer_sixdof_deckless_position_yaw_dagger_20260703.bin \
  --task position_yaw \
  --reset-profile broad \
  --sensor-profile deckless \
  --num-envs 768 \
  --collect-steps 512 \
  --hidden-size 128 \
  --num-layers 2 \
  --epochs 8 \
  --minibatch-size 8192 \
  --learning-rate 0.0003 \
  --seed 7305 \
  --dagger-iterations 4 \
  --dagger-steps 512 \
  --dagger-beta 0.0 \
  --previous-action-observation-scale 0.25 \
  --policy-envelope-coef 0.02 \
  --policy-action-abs-limit 0.98 \
  --no-wandb
```

Validate it before any hardware-facing use:

```bash
python scripts/evaluate_puffer_sixdof_checkpoint.py \
  --checkpoint artifacts/checkpoints/puffer_sixdof_deckless_position_yaw_dagger_20260703.bin \
  --task position_yaw \
  --backend python \
  --output artifacts/replay/puffer_sixdof_deckless_position_yaw_dagger_20260703_eval.json \
  --steps 500 \
  --num-envs 256 \
  --seed 7311 \
  --reset-profile broad \
  --sensor-profile deckless \
  --previous-action-observation-scale 0.25 \
  --min-completed-fraction 0.90 \
  --max-position-error-m 1.00 \
  --max-horizontal-speed-p95-m-s 2.0 \
  --max-tilt-p95-deg 35.0
```

The native PufferLib PPO lane is still useful for throughput experiments, but
the first deckless PPO run did not pass the offline gate. Keep it behind the
same evaluation command:

```bash
python scripts/train_sixdof_puffer4.py \
  --pufferlib-root /path/to/PufferLib \
  --sim-profile deckless \
  --task position_yaw \
  --reward-mode env \
  --reset-profile broad \
  --build-mode cpu \
  -- --train.total-timesteps 1048576
```

`deckless` keeps the fixed 28-value observation contract and fills the six
range slots with the existing max-range sentinel. Range-dependent tasks and
clearance reward modes are rejected for this profile so obstacle work stays
blocked until a ranger, camera, or external perception source is available.

## Hardware Logging

For real Crazyflie logging without the Multi-ranger deck, use:

```bash
python scripts/crazyflie_log.py \
  --config configs/hardware/crazyflie_2_1_brushless_flow_only.toml \
  --duration-s 20
```

This config still expects the Flow deck, but it does not expect the
Multi-ranger deck and does not request `range.*` log variables.

To exercise the policy path without taking control:

```bash
python scripts/crazyflie_sixdof_puffer_shadow_monitor.py \
  --checkpoint artifacts/checkpoints/puffer_sixdof_deckless_position_yaw_dagger_20260703.bin \
  --hardware-config configs/hardware/crazyflie_2_1_brushless_flow_only.toml \
  --output artifacts/crazyflie_logs/puffer_deckless_shadow.csv \
  --duration-s 12 \
  --previous-action-observation-scale 0.25
```

This command is monitor-only: it logs raw policy outputs and does not send
commands to the drone.

For a charged-drone baseline run, keep control with MotionCommander hover and
log Puffer outputs in shadow:

```bash
python scripts/crazyflie_baseline_puffer_shadow.py \
  --checkpoint artifacts/checkpoints/puffer_sixdof_deckless_position_yaw_dagger_20260703.bin \
  --hardware-config configs/hardware/crazyflie_2_1_brushless_flow_only.toml \
  --output artifacts/crazyflie_logs/baseline_puffer_shadow_20260703.csv \
  --duration-s 8 \
  --height-m 0.35 \
  --previous-action-observation-scale 0.25 \
  --confirm-flight
```

Rows from this runner include `baseline_controls_drone=True` and
`puffer_controls_drone=False`.

## Re-enabling Sensors

- Use `--sim-profile ranger` or a measured JSON sensor profile once the
  replacement Multi-ranger deck is working.
- Add a future AI-deck profile as a separate perception observation source
  instead of changing the deckless contract.
- Keep text or vision systems above the policy as structured mission goals and
  bounded setpoints; do not route them directly to motors.
