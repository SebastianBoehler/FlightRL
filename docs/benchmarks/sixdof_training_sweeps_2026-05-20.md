# 6-DoF Training Sweeps

Date: 2026-05-20

This note tracks training-loop experiments that feed the candidate matrix. It is simulation-only evidence, not live hardware approval.

## Loss Weights

Static imitation loss weights did not beat the baseline multi-task DAgger checkpoint. A 300-step screen kept `safe_horizon800` as best (`completed=0.5651`, `pos_err=2.8909m`, `clearance_p01=0.0535m`), while all loss-weight variants dropped completion to roughly `0.20-0.26`.

Conclusion: weighting existing samples is not enough; failing tasks need better visited-state coverage or a different closed-loop objective.

## Task Sampling

Teacher and DAgger collection now accept repeatable rollout sampling weights:

```bash
python scripts/build_sixdof_teacher_dataset.py \
  --task position_yaw,obstacle_avoidance,circle \
  --num-envs 12 \
  --steps 4 \
  --seed 901 \
  --task-probability position_yaw=2 \
  --task-probability circle=2 \
  --output artifacts/datasets/sixdof_task_probability_smoke.npz
```

Smoke result: metadata recorded sampling probabilities `position_yaw=0.4`, `obstacle_avoidance=0.2`, `circle=0.4`, with sampled counts `position_yaw=15`, `obstacle_avoidance=12`, `circle=21` across 48 samples.

DAgger CLI smoke:

```bash
python scripts/train_sixdof_dagger.py \
  --seed-dataset artifacts/datasets/sixdof_task_probability_smoke.npz \
  --initial-checkpoint artifacts/dagger/sixdof_safe_tasks_horizon800/iter_01.pt \
  --output-dir artifacts/dagger/task_probability_smoke \
  --iterations 1 \
  --num-envs 8 \
  --steps 2 \
  --epochs 1 \
  --batch-size 8 \
  --hidden-size 16 \
  --eval-steps 4 \
  --eval-num-envs 4 \
  --task-probability position_yaw=2 \
  --task-probability circle=2
```

Full one-iteration sampling sweep:

```bash
python scripts/run_sixdof_task_probability_sweep.py \
  --run \
  --baseline-checkpoint artifacts/dagger/sixdof_safe_tasks_horizon800/iter_01.pt \
  --report artifacts/replay/sixdof_task_probability_sweep_300_full.json \
  --output-dir artifacts/task_probability_sweep/suite300_full \
  --iterations 1 \
  --num-envs 128 \
  --steps 128 \
  --eval-steps 80 \
  --eval-num-envs 64 \
  --suite-steps 300 \
  --suite-num-envs 128
```

| variant | completed | pos err m | clearance p01 m |
| --- | ---: | ---: | ---: |
| baseline | 0.5651 | 2.8909 | 0.0535 |
| uniform_dagger | 0.2370 | 5.0093 | 0.0406 |
| sample_position_circle_2 | 0.2318 | 4.9852 | 0.0409 |
| sample_position_circle_3 | 0.2344 | 4.9854 | 0.0432 |
| sample_circle_3 | 0.2500 | 5.0535 | 0.0451 |
| sample_position_3 | 0.2370 | 4.9925 | 0.0411 |
| sample_position_circle_beta25 | 0.2500 | 5.1059 | 0.0394 |

Conclusion: one short DAgger iteration with altered task sampling still degrades the multi-task baseline.

## Closed-Loop PPO

`scripts/run_sixdof_ppo_sweep.py` now supports a baseline row plus runtime knobs for train env count, rollout horizon, minibatch size, train eval steps, gate eval envs, and medium/broad gate horizons.

Smoke command:

```bash
python scripts/run_sixdof_ppo_sweep.py \
  --run \
  --max-variants 1 \
  --baseline-checkpoint artifacts/curriculum/position_yaw/easy_medium_h128/checkpoint.pt \
  --init-checkpoint artifacts/curriculum/position_yaw/easy_medium_h128/checkpoint.pt \
  --report artifacts/replay/sixdof_position_yaw_ppo_sweep_smoke.json \
  --output-dir artifacts/ppo/position_yaw_smoke \
  --train-num-envs 8 \
  --horizon 4 \
  --minibatch-size 8 \
  --train-eval-steps 4 \
  --eval-num-envs 4 \
  --medium-steps 4 \
  --broad-steps 4 \
  --no-native-step
```

This wrote `artifacts/replay/sixdof_position_yaw_ppo_sweep_smoke.md` with both baseline and first PPO variant marked `ok`. The short gate horizons are command-path evidence only; the next quality screen should run longer native-step PPO variants.
