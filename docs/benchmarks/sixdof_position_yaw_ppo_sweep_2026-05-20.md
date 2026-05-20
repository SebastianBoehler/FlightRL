# 6-DoF Position/Yaw PPO Sweep

Date: 2026-05-20

Change: added `scripts/run_sixdof_ppo_sweep.py` to plan and execute conservative PPO tuning runs from the current best curriculum checkpoint. The sweep varies action standard deviation, teacher regularization, reference-policy regularization, and learning rate, then evaluates medium and broad reset gates.

Command:

```bash
python scripts/run_sixdof_ppo_sweep.py \
  --run \
  --max-variants 2 \
  --report artifacts/replay/sixdof_position_yaw_ppo_sweep_ref_smoke.json \
  --output-dir artifacts/ppo/position_yaw
```

Smoke result:

| variant | medium completed | medium survival | medium pos err m | broad completed | broad survival | broad pos err m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ref1_std006_lr5e5 | 0.5469 | 0.8647 | 3.2926 | 0.0195 | 0.3192 | 88.0026 |
| ref2_std006_lr5e5 | 0.5430 | 0.8638 | 3.2970 | 0.0195 | 0.3192 | 87.9308 |

Best ranked candidate:

- `ref1_std006_lr5e5` for medium and broad by gate-score ordering.

Conclusion: the PPO sweep runner is working, but these conservative settings still do not beat `curriculum_h128` on medium starts (`completed=0.6016`). Reference regularization at `1.0` and `2.0` behaves nearly identically for this short run. The next PPO tuning pass should explore reward scaling or advantage shaping rather than only increasing reference strength.

Label alignment fix: `collect_rollout` now stores teacher labels before applying the policy action, matching the recorded observation. The same two-variant sweep was rerun:

```bash
python scripts/run_sixdof_ppo_sweep.py \
  --run \
  --max-variants 2 \
  --report artifacts/replay/sixdof_position_yaw_ppo_sweep_aligned_labels.json \
  --output-dir artifacts/ppo/position_yaw_aligned
```

Aligned-label result:

| variant | medium completed | medium survival | medium pos err m | broad completed | broad survival | broad pos err m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ref1_std006_lr5e5 | 0.4727 | 0.8451 | 3.2287 | 0.0156 | 0.3187 | 85.8053 |
| ref2_std006_lr5e5 | 0.5430 | 0.8638 | 3.2971 | 0.0195 | 0.3192 | 87.9312 |

After label alignment, `ref2_std006_lr5e5` ranks best, but still fails all gates and remains below `curriculum_h128` on medium completion.

Progress reward shaping pass: PPO rollout collection now supports `--reward-mode progress`, which stores a shaped reward based on position-error progress, speed, yaw-error proxy, horizontal clearance, control effort, and termination. The base simulator reward remains available as `--reward-mode env`.

Command:

```bash
python scripts/run_sixdof_ppo_sweep.py \
  --run \
  --report artifacts/replay/sixdof_position_yaw_ppo_sweep_progress.json \
  --output-dir artifacts/ppo/position_yaw_progress
```

Progress-shaping result:

| variant | reward mode | medium completed | medium survival | medium pos err m | medium clearance p01 m | broad completed | broad survival | broad pos err m |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ref1_std006_lr5e5 | env | 0.4727 | 0.8451 | 3.2287 | 0.0512 | 0.0156 | 0.3187 | 85.8053 |
| ref2_std006_lr5e5 | env | 0.5430 | 0.8638 | 3.2971 | 0.0553 | 0.0195 | 0.3192 | 87.9312 |
| progress_ref1_std006 | progress | 0.5938 | 0.8817 | 2.9854 | 0.0688 | 0.0156 | 0.3201 | 84.2743 |
| progress_ref2_std004 | progress | 0.6172 | 0.8881 | 2.8228 | 0.0745 | 0.0156 | 0.3211 | 84.9585 |

Direct validation-suite comparison:

```bash
python scripts/evaluate_sixdof_suite.py \
  --teacher teacher_medium position_yaw \
  --candidate curriculum_h128 artifacts/curriculum/position_yaw/easy_medium_h128/checkpoint.pt position_yaw \
  --candidate ppo_progress_ref2 artifacts/ppo/position_yaw_progress/progress_ref2_std004/checkpoint.pt position_yaw \
  --steps 400 --num-envs 256 --native-step --reset-profile position_yaw_medium \
  --output artifacts/replay/sixdof_position_yaw_ppo_progress_medium_suite.json
```

| profile | candidate | completed | survival | pos err m | clearance p01 m | passed |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| medium | teacher_medium | 1.0000 | 1.0000 | 0.1354 | 1.0227 | yes |
| medium | curriculum_h128 | 0.6016 | 0.8767 | 3.1451 | 0.0712 | no |
| medium | ppo_progress_ref2 | 0.6289 | 0.8838 | 2.9349 | 0.0716 | no |
| broad | teacher_broad | 0.9844 | 0.9880 | 0.0625 | 0.4581 | yes |
| broad | curriculum_h128 | 0.0117 | 0.3296 | 88.7315 | 0.0421 | no |
| broad | ppo_progress_ref2 | 0.0234 | 0.3582 | 76.1579 | 0.0389 | no |

Conclusion: progress shaping is the best short PPO variant so far. It slightly beats `curriculum_h128` on the medium reset suite, but still fails clearance, completion, and position-error gates. Broad reset behavior remains weak, so the next training change should target curriculum breadth and obstacle clearance rather than just PPO optimizer knobs.

Broad clearance pass: PPO rollout rewards now also support `--reward-mode progress_clearance`, with stronger horizontal clearance pressure for room-scale starts. The PPO sweep includes two broad-reset variants.

```bash
python scripts/run_sixdof_ppo_sweep.py \
  --run \
  --report artifacts/replay/sixdof_position_yaw_ppo_sweep_broad_clearance.json \
  --output-dir artifacts/ppo/position_yaw_broad_clearance
```

| variant | train profile | reward mode | medium completed | broad completed | broad survival | broad pos err m | broad clearance p01 m |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| progress_ref2_std004 | position_yaw_medium | progress | 0.6172 | 0.0156 | 0.3211 | 84.9585 | 0.0377 |
| broad_clearance_ref2_std006 | broad | progress_clearance | 0.5195 | 0.0156 | 0.3438 | 80.2768 | 0.0389 |
| broad_clearance_ref1_std004 | broad | progress_clearance | 0.5391 | 0.0156 | 0.3226 | 90.2424 | 0.0405 |

One longer broad PPO fine-tune from the broad curriculum checkpoint was also run:

```bash
python scripts/train_sixdof_ppo.py \
  --init-checkpoint artifacts/curriculum/position_yaw/easy_medium_broad_h256/checkpoint.pt \
  --checkpoint artifacts/ppo/position_yaw_broad_clearance/long_broad_from_curriculum_broad/checkpoint.pt \
  --updates 64 --num-envs 1024 --horizon 64 --hidden-size 256 \
  --learning-rate 2e-5 --update-epochs 2 --minibatch-size 8192 \
  --action-std 0.05 --imitation-coef 0.10 --reference-coef 1.0 \
  --reward-mode progress_clearance --reset-profile broad --eval-reset-profile broad \
  --eval-steps 800 --eval-num-envs 256 --native-step
```

The saved selection was update `32`, with broad completion `0.0156`, survival `0.3016`, position error `93.9378m`, and clearance p01 `0.0360m`. The first evaluation during that run briefly reached completion `0.105`, then degraded, so longer broad PPO from the unstable broad imitation checkpoint is not a useful default yet.
