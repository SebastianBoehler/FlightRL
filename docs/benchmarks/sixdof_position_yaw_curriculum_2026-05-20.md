# 6-DoF Position/Yaw Curriculum Attempt

Date: 2026-05-20

Change: added named reset profiles for staged position/yaw data collection:

- `position_yaw_easy`: small target offsets, small yaw offsets, low attitude disturbance
- `position_yaw_medium`: larger but still local target/yaw offsets
- `broad`: previous random room reset distribution

Commands:

```bash
python scripts/build_sixdof_teacher_dataset.py \
  --task position_yaw --num-envs 512 --steps 192 --seed 809 \
  --native-step --reset-profile position_yaw_easy \
  --output artifacts/datasets/sixdof_teacher_position_yaw_easy_512x192.npz

python scripts/build_sixdof_teacher_dataset.py \
  --task position_yaw --num-envs 512 --steps 192 --seed 811 \
  --native-step --reset-profile position_yaw_medium \
  --append-dataset artifacts/datasets/sixdof_teacher_position_yaw_easy_512x192.npz \
  --output artifacts/datasets/sixdof_teacher_position_yaw_curriculum_easy_medium_512x384.npz

python scripts/train_sixdof_offline.py \
  --dataset artifacts/datasets/sixdof_teacher_position_yaw_curriculum_easy_medium_512x384.npz \
  --checkpoint artifacts/checkpoints/sixdof_position_yaw_curriculum_h256.pt \
  --epochs 12 --batch-size 8192 --hidden-size 256 \
  --select-by-eval --eval-reset-profile position_yaw_medium --native-step
```

Validation:

| suite | checkpoint | passed | pos err m | clearance p01 m | completed | survival |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| easy 400-step | curriculum offline | false | 1.4266 | 0.1648 | 0.8125 | 0.9467 |
| medium 400-step | curriculum offline | false | 3.1945 | 0.0727 | 0.6562 | 0.8754 |
| broad 800-step | curriculum offline | false | 93.1986 | 0.0395 | 0.0117 | 0.3137 |
| broad 800-step | obstacle focus | true | 0.1347 | 0.5448 | 1.0000 | 1.0000 |

DAgger with `--reset-profile position_yaw_medium --eval-reset-profile position_yaw_medium --beta 0.50` did not improve the medium gate. Iteration 1 reached `completed=0.3516`, `survival=0.7855`, and `pos_err=4.8083`.

Conclusion: curriculum reset profiles are now available and improve the position/yaw training signal, but imitation is still not a passing broad position/orientation controller. The next candidate should use an RL objective or closed-loop rollout loss instead of only teacher-action MSE.
