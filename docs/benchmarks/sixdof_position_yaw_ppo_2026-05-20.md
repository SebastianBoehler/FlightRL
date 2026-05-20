# 6-DoF Position/Yaw PPO Attempt

Date: 2026-05-20

Change: added a first closed-loop PPO-style training path for `SixDofCrazyflieEnv`. The trainer saves the actor as a normal `SixDofPolicy` checkpoint so existing gate, suite, and edge-export tooling can evaluate it.

Commands:

```bash
python scripts/train_sixdof_ppo.py \
  --init-checkpoint artifacts/curriculum/position_yaw/easy_medium_h128/checkpoint.pt \
  --checkpoint artifacts/checkpoints/sixdof_position_yaw_ppo_medium_h128.pt \
  --updates 16 --num-envs 512 --horizon 64 --hidden-size 128 \
  --action-std 0.18 --reset-profile position_yaw_medium \
  --eval-reset-profile position_yaw_medium --native-step

python scripts/train_sixdof_ppo.py \
  --init-checkpoint artifacts/curriculum/position_yaw/easy_medium_h128/checkpoint.pt \
  --checkpoint artifacts/checkpoints/sixdof_position_yaw_ppo_bc_medium_h128.pt \
  --updates 16 --num-envs 512 --horizon 64 --hidden-size 128 \
  --learning-rate 0.0002 --action-std 0.12 --imitation-coef 0.25 \
  --reset-profile position_yaw_medium --eval-reset-profile position_yaw_medium --native-step
```

Medium reset validation:

| checkpoint | passed | pos err m | clearance p01 m | completed | survival |
| --- | ---: | ---: | ---: | ---: | ---: |
| curriculum_h128 | false | 3.1451 | 0.0712 | 0.6016 | 0.8767 |
| ppo_medium_h128 | false | 7.9288 | 0.0553 | 0.4219 | 0.7567 |
| ppo_bc_medium_h128 | false | 3.9361 | 0.0508 | 0.4141 | 0.7997 |

Broad reset validation:

| checkpoint | passed | pos err m | clearance p01 m | completed | survival |
| --- | ---: | ---: | ---: | ---: | ---: |
| curriculum_h128 | false | 88.7315 | 0.0421 | 0.0117 | 0.3296 |
| ppo_bc_medium_h128 | false | 76.1963 | 0.0398 | 0.0117 | 0.3193 |

Conclusion: the closed-loop PPO scaffold works and produces gate-compatible checkpoints, but these first short runs do not improve position/yaw gates. Pure PPO drifts from the imitation policy, while policy-visited teacher regularization reduces position error but still hurts completion and clearance. The next iteration should tune reward scale/action variance or use a more conservative KL-style update before longer runs.
