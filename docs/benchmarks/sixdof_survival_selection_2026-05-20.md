# 6-DoF Survival-Based Selection Checkpoint

Date: 2026-05-20

Change: validation now reports `mean_survival_fraction`, and eval-selected offline/DAgger ranking uses it to distinguish failed checkpoints that nearly survive from ones that crash early.

Commands:

```bash
python scripts/evaluate_sixdof_suite.py \
  --teacher teacher_position position_yaw \
  --candidate position_dagger_iter2 artifacts/dagger/sixdof_position_yaw_eval_dagger/iter_02.pt position_yaw \
  --candidate obstacle_focus artifacts/dagger/sixdof_obstacle_focus_refine/iter_02.pt obstacle_avoidance \
  --steps 800 \
  --num-envs 256 \
  --native-step \
  --output artifacts/replay/sixdof_survival_metric_suite.json

python scripts/train_sixdof_dagger.py \
  --seed-dataset artifacts/datasets/sixdof_teacher_position_yaw_512x256.npz \
  --initial-checkpoint artifacts/checkpoints/sixdof_position_yaw_eval_selected_h256.pt \
  --output-dir artifacts/dagger/sixdof_position_yaw_survival_beta50 \
  --iterations 2 \
  --num-envs 512 \
  --steps 384 \
  --beta 0.50 \
  --task position_yaw \
  --epochs 10 \
  --select-by-eval \
  --native-step
```

Strict 800-step validation:

| label | passed | pos err m | clearance p01 m | completed | survival |
| --- | ---: | ---: | ---: | ---: | ---: |
| teacher_position | true | 0.0646 | 0.4001 | 0.9844 | 0.9883 |
| position_survival_beta50_iter1 | false | 51.2787 | 0.0415 | 0.0000 | 0.3280 |
| position_survival_beta50_iter2 | false | 258.7529 | 0.0942 | 0.0000 | 0.1043 |
| obstacle_focus | true | 0.1245 | 0.6007 | 1.0000 | 1.0000 |

Result: survival-aware selection provides a better ranking signal for failed position/yaw policies, but beta-0.50 DAgger is still not a passing position/orientation controller. The next position/yaw attempt should use a staged target-distance curriculum or an RL objective, not pure teacher imitation on broad random targets.
