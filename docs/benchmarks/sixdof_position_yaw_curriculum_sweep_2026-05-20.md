# 6-DoF Position/Yaw Curriculum Sweep

Date: 2026-05-20

Change: added `scripts/run_sixdof_curriculum_sweep.py` to plan and execute staged position/yaw curriculum sweeps. Each variant builds one or more teacher datasets, trains an offline checkpoint with eval-based selection, then writes medium and broad gate reports.

Smoke command:

```bash
python scripts/run_sixdof_curriculum_sweep.py \
  --run \
  --max-variants 1 \
  --report artifacts/replay/sixdof_position_yaw_curriculum_sweep_smoke.json \
  --output-dir artifacts/curriculum/position_yaw
```

Smoke result:

| variant | medium completed | medium survival | medium pos err m | broad completed | broad survival | broad pos err m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy_medium_h128 | 0.6289 | 0.8866 | 2.7011 | 0.0195 | 0.3237 | 85.7522 |

Artifacts:

- `artifacts/curriculum/position_yaw/easy_medium_h128/checkpoint.pt`
- `artifacts/curriculum/position_yaw/easy_medium_h128/medium_gate.json`
- `artifacts/curriculum/position_yaw/easy_medium_h128/broad_gate.json`
- `artifacts/replay/sixdof_position_yaw_curriculum_sweep_smoke.json`

Full sweep command:

```bash
python scripts/run_sixdof_curriculum_sweep.py \
  --run \
  --report artifacts/replay/sixdof_position_yaw_curriculum_sweep_full.json \
  --output-dir artifacts/curriculum/position_yaw
```

Full sweep result:

| variant | medium completed | medium survival | medium pos err m | broad completed | broad survival | broad pos err m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| easy_medium_h128 | 0.6289 | 0.8866 | 2.7011 | 0.0195 | 0.3237 | 85.7522 |
| easy_medium_h256 | 0.5586 | 0.8646 | 3.9850 | 0.0000 | 0.2936 | 99.7837 |
| medium_h256_long | 0.3164 | 0.6488 | 33.7290 | 0.0312 | 0.2917 | 134.2562 |
| easy_medium_broad_h256 | 0.3672 | 0.6669 | 24.3129 | 0.0547 | 0.3280 | 121.2577 |

Best ranked candidates:

- Medium reset profile: `easy_medium_h128`, still failed with `completion`, `min_clearance`, and `position_error`.
- Broad reset profile: `easy_medium_broad_h256`, still failed with `completion`, `min_clearance`, and `position_error`.

Conclusion: the sweep runner is working, but the tested imitation-only curriculum variants do not produce a passing position/orientation checkpoint. Adding broad teacher data slightly improves broad completion, but broad position error remains catastrophic. The next position/yaw path should add a closed-loop RL objective or rollout-loss term instead of widening the imitation dataset again.

Staged wide-profile bridge: added `position_yaw_wide` and `position_yaw_hard` reset profiles. These keep targets relative to the initial pose like the easy/medium curricula, but increase target distance, altitude offset, yaw offset, and attitude spread before jumping to fully random broad targets.

Manual command sequence:

```bash
python scripts/build_sixdof_teacher_dataset.py --task position_yaw --num-envs 512 --steps 192 --seed 701 --reset-profile position_yaw_easy --output artifacts/curriculum/position_yaw/easy_medium_wide_h128/dataset_01_position_yaw_easy.npz --native-step
python scripts/build_sixdof_teacher_dataset.py --task position_yaw --num-envs 512 --steps 192 --seed 702 --reset-profile position_yaw_medium --output artifacts/curriculum/position_yaw/easy_medium_wide_h128/dataset_02_position_yaw_medium.npz --append-dataset artifacts/curriculum/position_yaw/easy_medium_wide_h128/dataset_01_position_yaw_easy.npz --native-step
python scripts/build_sixdof_teacher_dataset.py --task position_yaw --num-envs 512 --steps 192 --seed 703 --reset-profile position_yaw_wide --output artifacts/curriculum/position_yaw/easy_medium_wide_h128/dataset_03_position_yaw_wide.npz --append-dataset artifacts/curriculum/position_yaw/easy_medium_wide_h128/dataset_02_position_yaw_medium.npz --native-step
python scripts/train_sixdof_offline.py --dataset artifacts/curriculum/position_yaw/easy_medium_wide_h128/dataset_03_position_yaw_wide.npz --checkpoint artifacts/curriculum/position_yaw/easy_medium_wide_h128/checkpoint.pt --epochs 14 --hidden-size 128 --learning-rate 8e-4 --eval-steps 500 --select-by-eval --eval-reset-profile position_yaw_wide --native-step
```

Staged wide result:

| profile | completed | survival | pos err m | clearance p01 m | passed |
| --- | ---: | ---: | ---: | ---: | --- |
| medium | 0.5781 | 0.8581 | 3.9816 | 0.0743 | no |
| wide | 0.3711 | 0.6900 | 16.4039 | 0.0488 | no |
| broad | 0.0352 | 0.3267 | 116.1111 | 0.0437 | no |

Conclusion: staged target-distance imitation is now represented in the code and artifacts, but the first wide checkpoint is worse than `easy_medium_h128` on medium and does not bridge to broad. This strengthens the case for a closed-loop rollout objective or recurrent/stateful policy rather than more static imitation data.
