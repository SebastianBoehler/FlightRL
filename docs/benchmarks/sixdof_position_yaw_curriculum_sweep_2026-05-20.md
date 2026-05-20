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

State-augmented observation pass: added `--observation-mode history1`, which trains/evaluates with current observation, one-step observation delta, and previous executed action. This is a feed-forward approximation to short memory that remains compatible with vectorized native/Puffer-style stepping.

Command:

```bash
python scripts/build_sixdof_teacher_dataset.py --task position_yaw --num-envs 512 --steps 192 --seed 711 --reset-profile position_yaw_easy --observation-mode history1 --output artifacts/curriculum/position_yaw/easy_medium_history1_h128/dataset_01_position_yaw_easy.npz --native-step
python scripts/build_sixdof_teacher_dataset.py --task position_yaw --num-envs 512 --steps 192 --seed 712 --reset-profile position_yaw_medium --observation-mode history1 --output artifacts/curriculum/position_yaw/easy_medium_history1_h128/dataset_02_position_yaw_medium.npz --append-dataset artifacts/curriculum/position_yaw/easy_medium_history1_h128/dataset_01_position_yaw_easy.npz --native-step
python scripts/train_sixdof_offline.py --dataset artifacts/curriculum/position_yaw/easy_medium_history1_h128/dataset_02_position_yaw_medium.npz --checkpoint artifacts/curriculum/position_yaw/easy_medium_history1_h128/checkpoint.pt --epochs 12 --hidden-size 128 --learning-rate 1e-3 --eval-steps 400 --eval-num-envs 256 --select-by-eval --eval-reset-profile position_yaw_medium --native-step
```

| profile | completed | survival | pos err m | clearance p01 m | passed |
| --- | ---: | ---: | ---: | ---: | --- |
| medium | 0.6250 | 0.8845 | 3.6545 | 0.0788 | no |
| broad | 0.0234 | 0.2786 | 109.1113 | 0.0500 | no |

Conclusion: `history1` improves medium clearance and keeps medium completion near the best static-imitation checkpoint, but broad survival regresses. It is useful scaffolding for future rollout/recurrent work, not a deployable position/yaw checkpoint.

Action-weighting ablation: added `--action-weighting inverse_std` to offline and DAgger training, plus a weighted curriculum variant in `run_sixdof_curriculum_sweep.py`. The intent was to make small control channels matter more than raw unweighted MSE.

Command:

```bash
python scripts/train_sixdof_offline.py \
  --dataset artifacts/curriculum/position_yaw/easy_medium_history1_h128/dataset_02_position_yaw_medium.npz \
  --checkpoint artifacts/checkpoints/sixdof_position_yaw_history1_weighted_h128.pt \
  --epochs 24 \
  --hidden-size 128 \
  --learning-rate 1e-3 \
  --eval-steps 500 \
  --eval-num-envs 256 \
  --select-by-eval \
  --eval-reset-profile position_yaw_medium \
  --native-step \
  --action-weighting inverse_std
```

| variant | medium completed | medium survival | medium pos err m | medium clearance p01 m | broad completed | broad survival | broad pos err m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| history1 control 24ep | 0.2703 | 0.6550 | 36.6386 | 0.0622 | 0.0312 | 0.2714 | 121.6190 |
| history1 inverse_std 24ep | 0.3672 | 0.7021 | 35.3566 | 0.0722 | 0.0742 | 0.3162 | 120.2756 |

Conclusion: inverse-std weighting slightly improves this same-budget control run, but both are much worse than the earlier `history1_h128` candidate. The weighting knob is useful for reproducible ablations, but it is not the missing ingredient for position/yaw. The next serious attempt should be closed-loop optimization or dataset aggregation that explicitly samples failure recovery states.

Recovery-state dataset pass: added `--execution-noise-std` to teacher dataset collection. Labels remain clean teacher commands, but the simulator executes clipped noisy teacher commands so the dataset includes off-trajectory recovery states. The `history1` observation stores the executed previous action.

Commands:

```bash
python scripts/build_sixdof_teacher_dataset.py --task position_yaw --num-envs 512 --steps 192 --seed 721 --reset-profile position_yaw_easy --observation-mode history1 --execution-noise-std 0.05 --output artifacts/curriculum/position_yaw/recovery_history1_h128/dataset_01_easy_noise.npz --native-step
python scripts/build_sixdof_teacher_dataset.py --task position_yaw --num-envs 512 --steps 192 --seed 722 --reset-profile position_yaw_medium --observation-mode history1 --execution-noise-std 0.05 --append-dataset artifacts/curriculum/position_yaw/recovery_history1_h128/dataset_01_easy_noise.npz --output artifacts/curriculum/position_yaw/recovery_history1_h128/dataset_02_medium_noise.npz --native-step
python scripts/train_sixdof_offline.py --dataset artifacts/curriculum/position_yaw/recovery_history1_h128/dataset_02_medium_noise.npz --checkpoint artifacts/checkpoints/sixdof_position_yaw_history1_recovery_h128.pt --epochs 16 --batch-size 8192 --hidden-size 128 --learning-rate 1e-3 --eval-steps 500 --eval-num-envs 256 --select-by-eval --eval-reset-profile position_yaw_medium --native-step
```

Mixed clean plus mild recovery command:

```bash
python scripts/build_sixdof_teacher_dataset.py --task position_yaw --num-envs 512 --steps 192 --seed 723 --reset-profile position_yaw_medium --observation-mode history1 --execution-noise-std 0.015 --append-dataset artifacts/curriculum/position_yaw/easy_medium_history1_h128/dataset_02_position_yaw_medium.npz --output artifacts/curriculum/position_yaw/mixed_recovery_history1_h128/dataset_clean_plus_medium_noise0015.npz --native-step
python scripts/train_sixdof_offline.py --dataset artifacts/curriculum/position_yaw/mixed_recovery_history1_h128/dataset_clean_plus_medium_noise0015.npz --checkpoint artifacts/checkpoints/sixdof_position_yaw_history1_mixed_recovery0015_h128.pt --epochs 14 --batch-size 8192 --hidden-size 128 --learning-rate 8e-4 --eval-steps 500 --eval-num-envs 256 --select-by-eval --eval-reset-profile position_yaw_medium --native-step
```

| variant | medium completed | medium survival | medium pos err m | medium clearance p01 m | broad completed | broad survival | broad pos err m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| history1 baseline | 0.6250 | 0.8845 | 3.6545 | 0.0788 | 0.0234 | 0.2786 | 109.1113 |
| noisy recovery 0.05 | 0.1484 | 0.6831 | 15.2497 | 0.0475 | 0.0156 | 0.3681 | 49.2559 |
| clean + recovery 0.015 | 0.5547 | 0.7926 | 21.3515 | 0.0763 | 0.1367 | 0.3762 | 91.3530 |

Conclusion: noisy teacher execution is now available for recovery-data ablations, but these two static-imitation variants do not pass the position/yaw gate. Mild mixed recovery improves broad completion versus the `history1` baseline, while medium position error regresses badly. This supports moving the next position/yaw effort toward closed-loop PPO/DAgger or a recurrent policy instead of more unweighted one-step behavior cloning.

Yaw gate propagation: the curriculum sweep now forwards the same yaw acceptance thresholds used by the checkpoint evaluator into every medium and broad gate command.

Dry-run manifest:

```bash
python scripts/run_sixdof_curriculum_sweep.py \
  --max-variants 1 \
  --report artifacts/replay/sixdof_position_yaw_curriculum_yaw_gated_manifest.json \
  --output-dir artifacts/curriculum/position_yaw_yaw_gated
```

The manifest records `max_yaw_error_rad = 0.35` and `max_yaw_p95_error_rad = 0.60`, and the generated eval commands include `--max-yaw-error-rad` plus `--max-yaw-p95-error-rad`. This keeps future offline/curriculum position-yaw checkpoints from ranking as candidates when they translate position but lose heading control.
