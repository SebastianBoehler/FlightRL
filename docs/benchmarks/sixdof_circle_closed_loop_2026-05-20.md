# 6-DoF Circle Closed-Loop Checkpoint Notes

Goal: train a simulation-only circle/orientation policy that can hold a circular trajectory with tangent yaw under the native 6-DoF simulator. This is not hardware-approved.

## Fixes

- Circle observations now encode tangent-yaw error instead of the reset `target_yaw`.
- Native-step circle rollouts recompute observations after the C-backed step.
- Circle reset profiles now initialize yaw near the tangent heading, with bounded recovery offsets.
- Offline eval-based checkpoint selection now records and scores yaw metrics:
  - `eval_yaw_error_rad`
  - `eval_yaw_error_p95_rad`
  - `mean_yaw_error_rad`
  - `yaw_error_p95_rad`
  - `action_saturation_fraction`
- DAgger iteration gates now accept yaw thresholds and rank yaw after survival/position.
- PPO now scores yaw metrics and uses orbit-radius error for circle progress reward instead of rewarding movement toward the circle center.
- Circle evaluation/gating now reports orbit-radius/height error as `mean_position_error_m` for the `circle` task instead of distance to the orbit center.
- Evaluation now supports `teacher_residual` checkpoints that execute `teacher_action + residual_scale * policy(obs)`. This creates a stable scaffold for residual learning without mislabeling it as a standalone learned policy.

## Evidence

Teacher controller, corrected circle recovery, 300 steps:

```bash
python scripts/evaluate_sixdof_checkpoint.py \
  --teacher \
  --task circle \
  --steps 300 \
  --num-envs 128 \
  --reset-profile circle_recovery \
  --output artifacts/curriculum/circle_aligned_h256_2026-05-20/teacher_circle_recovery_gate.json \
  --max-yaw-error-rad 0.35 \
  --max-yaw-p95-error-rad 0.6 \
  --native-step
```

Result: pass. Completion `0.9766`, position error `0.9386 m`, yaw mean `0.3193 rad`, yaw p95 `0.3764 rad`, clearance p01 `0.2390 m`.

After correcting circle position metrics to orbit error, the same teacher gate is:

- `artifacts/curriculum/circle_recovery_long_2026-05-20/teacher_circle_recovery_gate_orbit.json`
- Completion `0.9766`
- Orbit error `0.1831 m`
- Yaw mean `0.3193 rad`
- Yaw p95 `0.3764 rad`
- Clearance p01 `0.2390 m`

Best short-horizon BC checkpoint fails over 300 steps:

- `artifacts/curriculum/circle_aligned_smoke_2026-05-20/easy_recovery_h128/circle_recovery_300_gate.json`
- Completion `0.0938`
- Position error `5.6482 m`
- Yaw mean `1.5861 rad`
- Yaw p95 `2.8155 rad`

Larger BC checkpoint also fails:

- `artifacts/curriculum/circle_aligned_h256_2026-05-20/circle_recovery_gate.json`
- Completion `0.4219`
- Position error `3.6554 m`
- Yaw mean `0.8332 rad`
- Yaw p95 `1.6518 rad`

Inverse-std action weighting improved yaw mean in training history but failed closed-loop rollout:

- `artifacts/curriculum/circle_aligned_weighted_2026-05-20/circle_recovery_gate.json`
- Completion `0.2500`
- Position error `4.4006 m`
- Yaw mean `0.6212 rad`
- Yaw p95 `1.9769 rad`

DAgger did not recover a passing checkpoint in compact runs:

- `artifacts/dagger/circle_recovery_yaw_2026-05-20/summary.json`
- Iteration 2: completion `0.4297`, position error `3.5700 m`, yaw mean `0.6570 rad`, yaw p95 `1.7130 rad`
- `artifacts/dagger/circle_recovery_yaw_beta050_2026-05-20/summary.json`
- Iteration 1: completion `0.3438`, position error `3.8890 m`, yaw mean `0.3570 rad`, yaw p95 `1.1780 rad`

Long-horizon teacher dataset:

- `artifacts/datasets/sixdof_circle_recovery_long_512x300_2026-05-20.npz`
- 512 envs x 300 steps, native-step, circle recovery profile
- Terminal fraction `0.000033`

Long-horizon BC checkpoint improved but did not pass:

- `artifacts/curriculum/circle_recovery_long_2026-05-20/checkpoint.pt`
- `artifacts/curriculum/circle_recovery_long_2026-05-20/circle_recovery_gate.json`
- Completion `0.6172`
- Position error `2.8650 m`
- Yaw mean `0.7047 rad`
- Yaw p95 `1.5548 rad`
- Clearance p01 `0.0983 m`

With corrected orbit-error metrics:

- `artifacts/curriculum/circle_recovery_long_2026-05-20/circle_recovery_gate_orbit.json`
- Completion `0.6172`
- Orbit error `2.3197 m`
- Yaw mean `0.7047 rad`
- Yaw p95 `1.5548 rad`
- Clearance p01 `0.0983 m`

History-observation BC checkpoint:

- `artifacts/datasets/sixdof_circle_recovery_long_history1_512x300_2026-05-20.npz`
- `artifacts/curriculum/circle_recovery_long_history1_2026-05-20/checkpoint.pt`
- `artifacts/curriculum/circle_recovery_long_history1_2026-05-20/circle_recovery_gate_orbit.json`
- Completion `0.5938`
- Orbit error `2.9599 m`
- Yaw mean `0.4476 rad`
- Yaw p95 `1.0640 rad`
- Clearance p01 `0.0829 m`

PPO refinement from the long-horizon BC checkpoint did not improve enough:

- `artifacts/ppo/circle_recovery_long_orbit_yaw_2026-05-20/checkpoint.pt`
- `artifacts/ppo/circle_recovery_long_orbit_yaw_2026-05-20/circle_recovery_gate.json`
- Completion `0.5000`
- Position error `2.9025 m`
- Yaw mean `0.6958 rad`
- Yaw p95 `1.7695 rad`
- Clearance p01 `0.0698 m`

PPO refinement from the history-observation checkpoint also did not improve enough:

- `artifacts/ppo/circle_recovery_history1_orbit_yaw_2026-05-20/checkpoint.pt`
- `artifacts/ppo/circle_recovery_history1_orbit_yaw_2026-05-20/circle_recovery_gate_orbit.json`
- Completion `0.4844`
- Orbit error `3.1042 m`
- Yaw mean `0.5859 rad`
- Yaw p95 `1.5943 rad`
- Clearance p01 `0.0693 m`

Teacher-residual scaffold:

- `artifacts/residual/circle_teacher_residual_zero_2026-05-20/checkpoint.pt`
- `artifacts/residual/circle_teacher_residual_zero_2026-05-20/circle_recovery_gate.json`
- Controller: `teacher_residual`
- Residual scale: `0.0`
- Completion `0.9766`
- Orbit error `0.1831 m`
- Yaw mean `0.3193 rad`
- Yaw p95 `0.3764 rad`
- Teacher action L2 mean `0.0`
- Gate: pass

Zero-weight nonzero residual scaffold:

- `artifacts/residual/circle_teacher_residual_zero_weights_2026-05-20/checkpoint.pt`
- `artifacts/residual/circle_teacher_residual_zero_weights_2026-05-20/circle_recovery_gate.json`
- Controller: `teacher_residual`
- Residual scale: `0.1`
- Completion `0.9766`
- Orbit error `0.1831 m`
- Yaw mean `0.3193 rad`
- Yaw p95 `0.3764 rad`
- Teacher action L2 mean `0.0`
- Gate: pass

Compact PPO residual refinement:

- `artifacts/ppo/circle_residual_orbit_yaw_2026-05-20/checkpoint.pt`
- `artifacts/ppo/circle_residual_orbit_yaw_2026-05-20/checkpoint.report.json`
- `artifacts/ppo/circle_residual_orbit_yaw_2026-05-20/circle_recovery_gate.json`
- Controller: `teacher_residual`
- Residual scale: `0.1`
- Updates `8`, envs `256`, horizon `64`, native-step
- Completion `0.9766`
- Orbit error `0.1831 m`
- Yaw mean `0.3193 rad`
- Yaw p95 `0.3764 rad`
- Teacher action L2 mean `0.000006`
- Gate: pass

Residual PPO sweep automation:

- Script: `scripts/run_sixdof_residual_ppo_sweep.py`
- Smoke report: `artifacts/replay/sixdof_circle_residual_ppo_sweep_smoke_2026-05-20.json`
- Smoke markdown: `artifacts/replay/sixdof_circle_residual_ppo_sweep_smoke_2026-05-20.md`
- Smoke checkpoint: `artifacts/ppo/circle_residual_sweep_smoke_2026-05-20/scale005_ref4_std001/checkpoint.pt`
- Variant: `scale005_ref4_std001`
- Residual scale `0.05`, updates `1`, envs `32`, horizon `8`, native-step
- Completion `1.0000`
- Orbit error `0.1908 m`
- Yaw mean `0.1957 rad`
- Yaw p95 `0.4273 rad`
- Teacher action L2 mean `0.000002`
- Gate: pass

Smoke command:

```bash
python scripts/run_sixdof_residual_ppo_sweep.py \
  --run \
  --max-variants 1 \
  --updates 1 \
  --output-dir artifacts/ppo/circle_residual_sweep_smoke_2026-05-20 \
  --report artifacts/replay/sixdof_circle_residual_ppo_sweep_smoke_2026-05-20.json \
  --train-num-envs 32 \
  --horizon 8 \
  --minibatch-size 256 \
  --train-eval-steps 30 \
  --gate-steps 40 \
  --eval-num-envs 16 \
  --hidden-size 32 \
  --native-step
```

## Conclusion

The corrected teacher, zero-residual scaffold, and tiny-residual PPO checkpoint prove that residual learning can preserve the stable analytic controller in the simulator. Current standalone behavior cloning, history-observation behavior cloning, compact DAgger, and compact PPO policies are not robust enough for long-horizon circle/orientation control. Do not deploy standalone learned checkpoints to hardware.

Next useful step: run broader residual sweeps that explicitly trade reward against teacher-action deviation. Closed-loop distribution shift remains the current bottleneck for standalone policies.
