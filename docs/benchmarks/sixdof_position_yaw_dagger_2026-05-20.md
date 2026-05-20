# 6-DoF Position/Yaw DAgger Attempt

Date: 2026-05-20

Commands:

```bash
python scripts/build_sixdof_teacher_dataset.py \
  --task position_yaw \
  --num-envs 512 \
  --steps 256 \
  --seed 707 \
  --native-step \
  --output artifacts/datasets/sixdof_teacher_position_yaw_512x256.npz

python scripts/train_sixdof_offline.py \
  --dataset artifacts/datasets/sixdof_teacher_position_yaw_512x256.npz \
  --checkpoint artifacts/checkpoints/sixdof_position_yaw_eval_selected_h256.pt \
  --epochs 12 \
  --batch-size 8192 \
  --hidden-size 256 \
  --eval-steps 300 \
  --eval-num-envs 128 \
  --select-by-eval \
  --native-step

python scripts/train_sixdof_dagger.py \
  --seed-dataset artifacts/datasets/sixdof_teacher_position_yaw_512x256.npz \
  --initial-checkpoint artifacts/checkpoints/sixdof_position_yaw_eval_selected_h256.pt \
  --output-dir artifacts/dagger/sixdof_position_yaw_eval_dagger \
  --iterations 2 \
  --num-envs 512 \
  --steps 256 \
  --beta 0.25 \
  --task position_yaw \
  --epochs 8 \
  --eval-steps 300 \
  --eval-num-envs 128 \
  --select-by-eval \
  --native-step
```

Short-horizon DAgger result:

- Best checkpoint: `artifacts/dagger/sixdof_position_yaw_eval_dagger/iter_02.pt`
- 300-step completion: `0.2344`
- 300-step clearance p01 m: `0.0424`
- 300-step position error m: `4.1126`
- Gate: failed

Strict 800-step suite:

| label | passed | pos err m | clearance p01 m | completed | teacher L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| teacher_position | true | 0.0646 | 0.4001 | 0.9844 | 0.0000 |
| position_dagger_iter2 | false | 67.4896 | 0.0412 | 0.0000 | 0.8320 |
| position_offline | false | 56.9770 | 0.0398 | 0.0000 | 0.7298 |

Conclusion: position/yaw imitation remains unstable over longer horizons. The analytic teacher is sound, but the learned policy does not preserve enough closed-loop safety. The next attempt should change the task objective or train with a more conservative curriculum, not just add offline epochs.
