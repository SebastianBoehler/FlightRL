# 6-DoF Eval-Selected Offline Training

Date: 2026-05-20

The offline trainer now supports `--select-by-eval`, which chooses the saved checkpoint by closed-loop rollout metrics instead of validation MSE. The score prioritizes completion, then clearance, then position error, then validation loss.

Command:

```bash
python scripts/train_sixdof_offline.py \
  --dataset artifacts/datasets/sixdof_teacher_safe_tasks_512x256.npz \
  --checkpoint artifacts/checkpoints/sixdof_safe_tasks_eval_selected_safety_h256.pt \
  --epochs 8 \
  --batch-size 8192 \
  --hidden-size 256 \
  --learning-rate 0.001 \
  --eval-steps 300 \
  --eval-num-envs 128 \
  --select-by-eval \
  --native-step
```

Selection result:

- Selected epoch: `5`
- Validation loss: `0.0221997`
- Selection completion: `0.2292`
- Selection clearance p01 m: `0.0408`
- Selection position error m: `4.5672`

Candidate suite result:

| label | passed | pos err m | clearance p01 m | completed | teacher L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| eval_selected_safety | false | 4.7312 | 0.0349 | 0.2083 | 0.4934 |
| safe_horizon800 | false | 2.9872 | 0.0515 | 0.5182 | 0.3884 |
| obstacle_focus | true | 0.3994 | 0.4835 | 1.0000 | 0.0167 |

Conclusion: eval-based selection is useful infrastructure, but pure offline teacher imitation is still not enough for safe multi-task control. The next multi-task attempt should use stronger on-policy DAgger/curriculum or separate task experts, not only more offline epochs.
