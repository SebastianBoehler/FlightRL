# 6-DoF Multitask PPO Sweep

This note tracks the reusable profile-gated multitask PPO runner:

```bash
python scripts/run_sixdof_multitask_ppo_sweep.py
```

The runner trains from the safe multitask checkpoint, keeps the checkpoint task set as `position_yaw,obstacle_avoidance,circle`, validates all candidates across `position_yaw_recovery` and `broad`, then builds a profile matrix.

## Smoke Run

Command:

```bash
python scripts/run_sixdof_multitask_ppo_sweep.py \
  --run \
  --max-variants 1 \
  --updates 1 \
  --train-num-envs 64 \
  --horizon 16 \
  --minibatch-size 512 \
  --train-eval-steps 40 \
  --eval-num-envs 32 \
  --suite-steps 80 \
  --suite-num-envs 32 \
  --output-dir artifacts/ppo/multitask_profile_sweep_smoke_2026-05-20 \
  --report artifacts/replay/sixdof_multitask_ppo_sweep_smoke_2026-05-20.json
```

Artifacts:

- `artifacts/replay/sixdof_multitask_ppo_sweep_smoke_2026-05-20.json`
- `artifacts/replay/sixdof_multitask_ppo_sweep_smoke_2026-05-20.md`
- `artifacts/ppo/multitask_profile_sweep_smoke_2026-05-20/profile_matrix.json`
- `artifacts/ppo/multitask_profile_sweep_smoke_2026-05-20/balanced_h64_ref2_std002/checkpoint.pt`

| candidate | all passed | worst completed | worst pos err m | worst yaw rad | worst clearance m |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | false | 1.0000 | 0.7137 | 1.2741 | 0.8913 |
| balanced_h64_ref2_std002 | false | 1.0000 | 0.7493 | 1.2801 | 0.8453 |

Checkpoint metadata check:

- Hidden size inherited from the safe checkpoint: `256`.
- Tasks: `position_yaw`, `obstacle_avoidance`, `circle`.
- Task sampling: uniform.

Conclusion: the runner is operational, but the one-update smoke does not improve the baseline and remains yaw-gate blocked. The next useful run is a longer conservative PPO sweep, not promotion of this smoke checkpoint.

## Short Two-Variant Sweep

Command:

```bash
python scripts/run_sixdof_multitask_ppo_sweep.py \
  --run \
  --max-variants 2 \
  --updates 4 \
  --train-num-envs 128 \
  --horizon 64 \
  --minibatch-size 2048 \
  --train-eval-steps 120 \
  --eval-num-envs 64 \
  --suite-steps 180 \
  --suite-num-envs 64 \
  --output-dir artifacts/ppo/multitask_profile_sweep_short_2026-05-20 \
  --report artifacts/replay/sixdof_multitask_ppo_sweep_short_2026-05-20.json
```

| candidate | all passed | worst completed | worst pos err m | worst yaw rad | worst clearance m |
| --- | ---: | ---: | ---: | ---: | ---: |
| py_focus4_h64_ref2_std002 | false | 0.8802 | 1.1075 | 0.9014 | 0.1136 |
| balanced_h64_ref2_std002 | false | 0.8542 | 1.1203 | 0.8934 | 0.1004 |
| baseline | false | 0.8906 | 1.2512 | 0.9513 | 0.0910 |

Result: the position-yaw focused PPO continuation improved worst position error and clearance against baseline, but completion regressed slightly and yaw is still far outside the gate. This is a useful direction for longer training, not a deployable checkpoint.

## Task-Specific Yaw Reference

The yaw gate initially compared every task against `env.target_yaw`. That was wrong for `circle`: the circle teacher points yaw along the tangent direction instead of the reset target yaw. The shared task yaw reference now uses tangent yaw for `circle` in the teacher, PPO reward, and evaluation.

Regression tests:

```bash
pytest tests/test_sixdof_yaw.py tests/test_sixdof_rl.py tests/test_sixdof_evaluation.py tests/test_sixdof_multitask_ppo_sweep.py
```

Result: `16 passed`.

Task-aware yaw sweep:

```bash
python scripts/run_sixdof_multitask_ppo_sweep.py \
  --run \
  --max-variants 3 \
  --updates 4 \
  --train-num-envs 128 \
  --horizon 64 \
  --minibatch-size 2048 \
  --train-eval-steps 120 \
  --eval-num-envs 64 \
  --suite-steps 180 \
  --suite-num-envs 64 \
  --output-dir artifacts/ppo/multitask_profile_sweep_task_yaw_2026-05-20 \
  --report artifacts/replay/sixdof_multitask_ppo_sweep_task_yaw_2026-05-20.json
```

| candidate | all passed | worst completed | worst pos err m | worst yaw rad | worst clearance m |
| --- | ---: | ---: | ---: | ---: | ---: |
| py_focus4_h64_ref2_std002 | false | 0.8802 | 1.1156 | 0.4448 | 0.1113 |
| balanced_h64_ref2_std002 | false | 0.8646 | 1.1101 | 0.4575 | 0.0984 |
| baseline | false | 0.8906 | 1.2512 | 0.4367 | 0.0910 |
| py_yaw_focus4_h64_ref2_std002 | false | 0.8594 | 1.2896 | 0.4713 | 0.0956 |

Result: task-specific yaw roughly halves the old worst yaw metric, proving the previous gate was too pessimistic for `circle`. The remaining blockers are completion, position error, and yaw p95. The yaw-focused PPO variant did not beat the position-yaw focused variant in this short run.

## Per-Task Profile Matrix

`scripts/build_sixdof_profile_matrix.py` now writes `task_records` and renders a `Per-Task Blockers` table. This keeps the profile gate actionable for multitask checkpoints by showing which task is causing the aggregate failure.

Rebuilt matrix:

```bash
python scripts/build_sixdof_profile_matrix.py \
  --suite artifacts/ppo/multitask_profile_sweep_task_yaw_2026-05-20/profile_position_yaw_recovery.json \
  --suite artifacts/ppo/multitask_profile_sweep_task_yaw_2026-05-20/profile_broad.json \
  --output artifacts/ppo/multitask_profile_sweep_task_yaw_2026-05-20/profile_matrix.json
```

Top blockers:

| candidate | task | worst completed | worst pos err m | worst yaw rad | yaw p95 rad | worst clearance m |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| py_focus4_h64_ref2_std002 | circle | 0.7031 | 1.7110 | 0.9003 | 2.7808 | 0.1113 |
| balanced_h64_ref2_std002 | circle | 0.7031 | 1.6222 | 0.9254 | 2.8667 | 0.0984 |
| py_yaw_focus4_h64_ref2_std002 | circle | 0.7344 | 2.1865 | 0.9961 | 2.9072 | 0.0956 |
| baseline | circle | 0.7344 | 2.1433 | 0.7975 | 2.7177 | 0.0910 |

Conclusion: the next training target is not generic multitask PPO. The circle task needs a better curriculum or teacher/dataset treatment first, because it dominates completion, position error, and yaw p95 failures.
