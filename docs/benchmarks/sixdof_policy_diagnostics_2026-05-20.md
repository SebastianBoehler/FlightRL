# 6-DoF Policy Diagnostics

Date: 2026-05-20

Scope: offline simulator diagnostics only. No live Crazyflie hardware commands were run.

## Commands

```bash
python scripts/diagnose_sixdof_policy.py \
  --checkpoint artifacts/curriculum/position_yaw/easy_medium_history1_h128/checkpoint.pt \
  --task position_yaw \
  --profiles position_yaw_easy position_yaw_medium position_yaw_wide broad \
  --steps 400 \
  --num-envs 256 \
  --output artifacts/replay/sixdof_position_yaw_history1_diagnostics.json
```

```bash
python scripts/diagnose_sixdof_policy.py \
  --teacher \
  --task position_yaw \
  --profiles position_yaw_easy position_yaw_medium position_yaw_wide broad \
  --steps 400 \
  --num-envs 256 \
  --output artifacts/replay/sixdof_position_yaw_teacher_diagnostics.json
```

```bash
python scripts/diagnose_sixdof_policy.py \
  --checkpoint artifacts/dagger/sixdof_safe_tasks_horizon800/iter_01.pt \
  --task position_yaw \
  --profiles position_yaw_easy position_yaw_medium broad \
  --steps 400 \
  --num-envs 256 \
  --output artifacts/replay/sixdof_multitask_position_yaw_diagnostics.json
```

Additional multitask diagnostics:

- `artifacts/replay/sixdof_multitask_obstacle_diagnostics.json`
- `artifacts/replay/sixdof_multitask_circle_diagnostics.json`

## Results

| controller | task | profile | survival | pos err m | clearance p01 m | yaw err rad | blocker |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| history1_h128 | position_yaw | easy | 0.8398 | 1.5953 | 0.1739 | 0.0456 | survival |
| history1_h128 | position_yaw | medium | 0.6211 | 3.9928 | 0.0609 | 0.1712 | survival |
| history1_h128 | position_yaw | wide | 0.3438 | 8.6681 | 0.0337 | 0.3623 | survival |
| history1_h128 | position_yaw | broad | 0.1172 | 20.7654 | 0.1003 | 0.6871 | survival |
| teacher | position_yaw | broad | 0.9961 | 0.3880 | 1.1976 | 0.0043 | none |
| safe_horizon800 | position_yaw | easy | 0.9727 | 0.7278 | 0.0653 | 0.0533 | clearance |
| safe_horizon800 | position_yaw | medium | 0.8945 | 1.1882 | 0.0211 | 0.0898 | survival |
| safe_horizon800 | obstacle | broad | 0.6094 | 3.1525 | 0.0320 | 0.3158 | survival |
| safe_horizon800 | circle | broad | 0.2148 | 8.8741 | 0.0553 | 1.8993 | survival |

## Implication

The analytic teacher solves the same position-yaw profiles, so the current learned checkpoints are the blocker rather than the environment. The next training iteration should prioritize survival and clearance on easy/medium position-yaw before broadening the curriculum. Multitask training should not be promoted until the single-task position-yaw gate reaches stable survival; the current multitask candidate improves easy position error but trades away clearance and broad robustness.

## PPO Survival Screen

```bash
python scripts/run_sixdof_ppo_sweep.py \
  --run \
  --max-variants 2 \
  --init-checkpoint artifacts/curriculum/position_yaw/easy_medium_history1_h128/checkpoint.pt \
  --baseline-checkpoint artifacts/curriculum/position_yaw/easy_medium_history1_h128/checkpoint.pt \
  --output-dir artifacts/ppo/position_yaw_survival_screen \
  --report artifacts/replay/sixdof_position_yaw_survival_screen.json \
  --train-num-envs 128 \
  --horizon 32 \
  --minibatch-size 2048 \
  --train-eval-steps 160 \
  --eval-num-envs 64 \
  --medium-steps 200 \
  --broad-steps 300 \
  --native-step
```

| variant | medium pass | medium completed | broad completed | broad pos err m |
| --- | ---: | ---: | ---: | ---: |
| baseline | true | 0.9062 | 0.2031 | 9.7991 |
| stable_ref4_std002_lr1e5 | false | 0.8906 | 0.2344 | 10.2090 |
| stable_ref8_std001_lr5e6 | true | 0.9062 | 0.2344 | 9.8919 |

The conservative PPO variants do not materially fix broad survival and can degrade medium position error. The next useful experiment is not more tiny reference-anchored PPO from this checkpoint; it should add stronger recovery/safety data or restart position-yaw training with teacher-rich recovery states before broad multitask training.

## Recovery Dataset Refresh

Added `position_yaw_recovery`, a reset profile with wider target offsets and stronger initial attitude perturbations. This profile is meant to expose recovery behavior before any live-hardware policy test.

```bash
python scripts/build_sixdof_teacher_dataset.py \
  --task position_yaw \
  --num-envs 512 \
  --steps 192 \
  --seed 812 \
  --reset-profile position_yaw_recovery \
  --observation-mode history1 \
  --execution-noise-std 0.08 \
  --output artifacts/datasets/sixdof_position_yaw_recovery_history1_512x192_noise008.npz \
  --native-step
```

```bash
python scripts/train_sixdof_offline.py \
  --dataset artifacts/datasets/sixdof_position_yaw_recovery_history1_512x192_noise008.npz \
  --checkpoint artifacts/checkpoints/sixdof_position_yaw_recovery_history1_512x192_noise008_h128.pt \
  --epochs 10 \
  --batch-size 8192 \
  --hidden-size 128 \
  --learning-rate 8e-4 \
  --eval-steps 240 \
  --eval-num-envs 128 \
  --select-by-eval \
  --eval-reset-profile position_yaw_recovery \
  --native-step
```

The dataset has `98,304` samples, `history1` observations, noisy-teacher execution, and near-zero terminal fraction (`0.000041`). Training reached validation loss `0.000918`.

| controller | profile | survival | pos err m | clearance p01 m | yaw err rad | blocker |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| teacher | position_yaw_recovery | 0.9766 | 0.2877 | 0.8348 | 0.0036 | none |
| recovery_h128_noise008 | position_yaw_easy | 0.6992 | 2.1574 | 0.0336 | 0.0408 | survival |
| recovery_h128_noise008 | position_yaw_medium | 0.4375 | 4.0923 | 0.0561 | 0.1188 | survival |
| recovery_h128_noise008 | position_yaw_recovery | 0.1055 | 10.1923 | 0.0248 | 0.5038 | survival |
| recovery_h128_noise008 | broad | 0.1680 | 7.5908 | 0.0142 | 0.3036 | survival |

Conclusion: larger noisy behavior cloning improves over the 8k-sample smoke run but is still not hardware-ready. The next useful policy work is closed-loop DAgger/PPO from recovery states, not direct deployment of this checkpoint.

## Closed-Loop DAgger Smoke

Added `scripts/run_sixdof_recovery_dagger_sweep.py` to run recovery-focused DAgger variants and immediately diagnose the resulting checkpoints on `easy`, `medium`, `recovery`, and `broad` reset profiles.

```bash
python scripts/run_sixdof_recovery_dagger_sweep.py \
  --run \
  --max-variants 2 \
  --num-envs 128 \
  --steps 96 \
  --epochs 4 \
  --eval-steps 160 \
  --eval-num-envs 128 \
  --diagnostic-steps 260 \
  --diagnostic-num-envs 128 \
  --report artifacts/replay/sixdof_position_yaw_recovery_dagger_smoke2.json \
  --output-dir artifacts/dagger/position_yaw_recovery_smoke2
```

| controller | profile | survival | pos err m | clearance p01 m | yaw err rad | blocker |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| recovery_h128_noise008 | position_yaw_recovery | 0.1055 | 10.1923 | 0.0248 | 0.5038 | survival |
| DAgger beta0.00 | position_yaw_recovery | 0.2188 | 5.3736 | 0.0283 | 0.2515 | survival |
| DAgger beta0.10 weighted | position_yaw_recovery | 0.2188 | 5.1315 | 0.1537 | 0.1778 | survival |
| DAgger beta0.10 weighted | broad | 0.4219 | 2.8604 | 0.0291 | 0.1693 | survival |

Conclusion: closed-loop DAgger roughly doubles recovery survival and improves broad survival versus the recovery BC checkpoint, but it is still below the safety gate. Continue with larger DAgger rollouts and/or PPO refinement from the weighted DAgger candidate before considering any live learned policy test.

## Profile Matrix Gate

Added `scripts/build_sixdof_profile_matrix.py` to aggregate multiple validation-suite reports by candidate and reset profile. This prevents a checkpoint from looking better because it was evaluated on only one profile.

```bash
python scripts/evaluate_sixdof_suite.py \
  --candidate recovery_bc artifacts/checkpoints/sixdof_position_yaw_recovery_history1_512x192_noise008_h128.pt position_yaw \
  --candidate recovery_dagger_beta010 artifacts/dagger/position_yaw_recovery_smoke2/policy_states_beta010_weighted/iter_02.pt position_yaw \
  --steps 300 \
  --num-envs 128 \
  --native-step \
  --reset-profile position_yaw_recovery \
  --max-yaw-error-rad 0.35 \
  --max-yaw-p95-error-rad 0.60 \
  --output artifacts/replay/sixdof_recovery_candidates_profile_recovery.json
```

The same command was run for `--reset-profile broad`, then aggregated:

```bash
python scripts/build_sixdof_profile_matrix.py \
  --suite artifacts/replay/sixdof_recovery_candidates_profile_recovery.json \
  --suite artifacts/replay/sixdof_recovery_candidates_profile_broad.json \
  --output artifacts/replay/sixdof_recovery_profile_matrix.json
```

| candidate | all profiles passed | worst survival | worst completed | worst pos err m | worst yaw rad | worst clearance m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| recovery_bc | false | 0.7360 | 0.3359 | 3.8862 | 0.1081 | 0.0321 |
| recovery_dagger_beta010 | false | 0.6071 | 0.0859 | 6.8729 | 0.2887 | 0.0285 |

Conclusion: the DAgger smoke improvement is not robust under the stricter yaw-gated suite and different evaluation seed. The current best recovery candidate by profile matrix is still the BC checkpoint, and neither candidate is promotable.

## Readiness Gate Integration

`scripts/build_sixdof_readiness_report.py` now accepts `--profile-matrix`. If profile evidence is supplied, position/yaw and multitask candidates containing `position_yaw` must have matching profile-matrix evidence and must pass every profile in it.

```bash
python scripts/build_sixdof_readiness_report.py \
  --matrix artifacts/replay/sixdof_candidate_matrix_current.json \
  --profile-matrix artifacts/replay/sixdof_recovery_profile_matrix.json \
  --room-report artifacts/replay/room_scan_autonomous_35s.clean20.strict_path.room.json \
  --native-parity artifacts/replay/sixdof_native_parity_current.json \
  --output artifacts/replay/sixdof_readiness_with_profile_matrix.json
```

Result: `obstacle_focus` remains readiness-green for the simulation/edge gate. `history1_h128` and `safe_horizon800` are now blocked by both `sim_gate` and `profile_matrix_missing`, because the current recovery/broad profile matrix only covers the recovery BC and recovery DAgger candidates. This is the intended promotion boundary: no position/yaw checkpoint should be treated as a readiness candidate without profile-level recovery/broad evidence.

## Current Candidate Profile Gate

Added `scripts/run_sixdof_profile_gate.py` to build profile evidence directly from the current candidate matrix. It selects readiness candidates that include `position_yaw`, evaluates them across the requested reset profiles, and writes the profile matrix consumed by the readiness report.

```bash
python scripts/run_sixdof_profile_gate.py \
  --run \
  --matrix artifacts/replay/sixdof_candidate_matrix_current.json \
  --profiles position_yaw_recovery broad \
  --steps 300 \
  --num-envs 128 \
  --output-dir artifacts/replay/profile_gate_current_candidates \
  --output artifacts/replay/sixdof_profile_gate_current_candidates.json \
  --profile-matrix-output artifacts/replay/sixdof_profile_matrix_current_candidates.json
```

| candidate | all profiles passed | worst survival | worst completed | worst pos err m | worst yaw rad | worst clearance m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| safe_horizon800 | false | 0.8664 | 0.5755 | 2.4123 | 0.8101 | 0.0524 |
| history1_h128 | false | 0.6528 | 0.2266 | 10.1018 | 0.5594 | 0.0413 |

Using this matrix in readiness replaces `profile_matrix_missing` with concrete `profile_matrix` failures for `history1_h128` and `safe_horizon800`. Obstacle avoidance remains the only readiness-green learned checkpoint in the current stack.

## Profile-Gated PPO Refine

Before task-conditioned PPO was available, a bounded profile-refine run from the compatible single-task `history1_h128` checkpoint wrote `artifacts/ppo/position_yaw_profile_refine/history1_recovery_ref2_std004.pt`.

| candidate | all profiles passed | worst survival | worst completed | worst pos err m | worst yaw rad | worst clearance m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| history1_recovery_ppo_ref2 | false | 0.6325 | 0.1797 | 11.2148 | 0.4980 | 0.0375 |
| history1_h128 | false | 0.6122 | 0.1875 | 11.6698 | 0.6482 | 0.0343 |

Result: this PPO refine slightly improves survival, position error, yaw, and clearance versus the baseline on the same profile matrix, but it lowers worst completion and remains far below the gate. It is an informative failed ablation, not a promotable checkpoint.

## Task-Conditioned PPO Smoke

PPO rollout collection and `train_sixdof_ppo.py` now support task-conditioned checkpoints. The first smoke from `safe_horizon800` wrote `artifacts/ppo/multitask_profile_refine/safe_horizon800_ref1_std003.pt`.

| candidate | all profiles passed | worst survival | worst completed | worst pos err m | worst yaw rad | worst clearance m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| safe_horizon800_ppo_ref1 | false | 0.8566 | 0.5521 | 2.5415 | 0.8607 | 0.0433 |
| safe_horizon800 | false | 0.8541 | 0.5651 | 2.8909 | 0.8429 | 0.0535 |

Result: task-conditioned PPO is now a working training path, but this short conservative refine is not promotable. It improves worst position error while regressing completion, yaw, and clearance.

The PPO trainer also accepts repeatable `--task-probability TASK=WEIGHT` flags. A position-yaw weighted smoke used `position_yaw=4`, `obstacle_avoidance=1`, `circle=1`:

| candidate | all profiles passed | worst survival | worst completed | worst pos err m | worst yaw rad | worst clearance m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| safe_horizon800_pyw4 | false | 0.8582 | 0.5495 | 2.4615 | 0.8250 | 0.0356 |
| safe_horizon800_ppo_ref1 | false | 0.8566 | 0.5521 | 2.5415 | 0.8607 | 0.0433 |
| safe_horizon800 | false | 0.8541 | 0.5651 | 2.8909 | 0.8429 | 0.0535 |

The weighted run improves position error and yaw compared with the uniform PPO refine, but completion and clearance still regress. Keep it as a tuning knob, not a promoted checkpoint.

## Task-Probability DAgger Refresh

The DAgger task-probability runner was exercised with a short quality sweep after the throughput refresh:

```bash
python scripts/run_sixdof_task_probability_sweep.py \
  --run \
  --max-variants 3 \
  --iterations 1 \
  --num-envs 128 \
  --steps 128 \
  --eval-steps 80 \
  --eval-num-envs 64 \
  --suite-steps 240 \
  --suite-num-envs 96 \
  --baseline-checkpoint artifacts/dagger/sixdof_safe_tasks_horizon800/iter_01.pt \
  --output-dir artifacts/task_probability_sweep/heartbeat_2026-05-20 \
  --report artifacts/replay/sixdof_task_probability_sweep_heartbeat_2026-05-20.json
```

| variant | probabilities | completed | pos err m | clearance p01 m |
| --- | --- | ---: | ---: | ---: |
| baseline | uniform | 0.6701 | 1.7181 | 0.0717 |
| uniform_dagger | uniform | 0.4375 | 2.8653 | 0.0624 |
| sample_position_circle_2 | position_yaw=2, circle=2 | 0.4340 | 2.8558 | 0.0613 |
| sample_position_circle_3 | position_yaw=3, circle=3 | 0.4340 | 2.8550 | 0.0586 |

Conclusion: this short DAgger continuation regressed against the safe multitask baseline. Do not promote these checkpoints; the next useful policy experiment is either longer PPO from the baseline or a dataset-quality change, not another one-iteration DAgger continuation.
