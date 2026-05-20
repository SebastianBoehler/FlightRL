# 6-DoF Candidate Screen

Date: 2026-05-20

Short native-step screen:

```bash
python scripts/evaluate_sixdof_suite.py \
  --teacher teacher_safe position_yaw,obstacle_avoidance,circle \
  --candidate safe_offline artifacts/checkpoints/sixdof_safe_tasks_offline_h256.pt checkpoint \
  --candidate safe_dagger_iter3 artifacts/dagger/sixdof_safe_tasks_iter3/iter_03.pt checkpoint \
  --candidate safe_horizon800 artifacts/dagger/sixdof_safe_tasks_horizon800/iter_01.pt checkpoint \
  --candidate safe_multitask artifacts/checkpoints/sixdof_safe_multitask_h256.pt checkpoint \
  --candidate obstacle_focus artifacts/dagger/sixdof_obstacle_focus_refine/iter_02.pt obstacle_avoidance \
  --steps 300 \
  --num-envs 128 \
  --native-step \
  --output artifacts/replay/sixdof_candidate_suite_safe_tasks.json
```

Result: only `obstacle_focus` passed. The broad safe-task checkpoints failed clearance, completion, and position-error gates.

Per-task split:

| task | best candidate | passed | pos err m | completed | clearance p01 m |
| --- | --- | ---: | ---: | ---: | ---: |
| position_yaw | safe_horizon800_position | false | 1.9459 | 0.6250 | 0.0841 |
| obstacle_avoidance | safe_dagger_iter3_obstacle | false | 1.8365 | 0.7344 | 0.1282 |
| circle | safe_horizon800_circle | false | 4.9263 | 0.3594 | 0.0446 |

Strict obstacle-specific suite:

```bash
python scripts/evaluate_sixdof_suite.py \
  --teacher teacher_obstacle obstacle_avoidance \
  --candidate obstacle_focus artifacts/dagger/sixdof_obstacle_focus_refine/iter_02.pt obstacle_avoidance \
  --steps 800 \
  --num-envs 256 \
  --native-step \
  --output artifacts/replay/sixdof_obstacle_focus_strict_suite.json
```

| label | passed | pos err m | clearance p01 m | completed | teacher L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| teacher_obstacle | true | 0.1001 | 0.5741 | 1.0000 | 0.0000 |
| obstacle_focus | true | 0.1402 | 0.4839 | 1.0000 | 0.0129 |

Conclusion: `artifacts/dagger/sixdof_obstacle_focus_refine/iter_02.pt` is the current strongest learned 6-DoF checkpoint. Multi-task position/yaw/circle behavior still needs more training or task objective changes before it should be considered deployment-adjacent.

Candidate matrix command:

```bash
python scripts/build_sixdof_candidate_matrix.py \
  --suite artifacts/replay/sixdof_candidate_suite_safe_tasks.json \
  --suite artifacts/replay/sixdof_obstacle_focus_strict_suite.json \
  --suite artifacts/replay/sixdof_position_yaw_ppo_progress_medium_suite.json \
  --suite artifacts/replay/sixdof_history1_medium_suite.json \
  --parity obstacle_focus=artifacts/edge/sixdof_obstacle_focus_refine.parity.json \
  --parity history1_h128=artifacts/edge/sixdof_history1_h128.parity.json \
  --latency obstacle_focus=artifacts/edge/sixdof_obstacle_focus_refine.latency.json \
  --latency history1_h128=artifacts/edge/sixdof_history1_h128.latency.json \
  --output artifacts/replay/sixdof_candidate_matrix_current.json
```

Current matrix result:

| task | selected label | passed | edge parity | latency us | completed | pos err m | clearance p01 m |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| obstacle_avoidance | obstacle_focus | true | true | 9.967 | 1.0000 | 0.1402 | 0.4839 |
| position_yaw | history1_h128 | false | true | 9.323 | 0.6602 | 3.8810 | 0.0889 |

The matrix makes the current boundary explicit: obstacle avoidance has a learned checkpoint with gate pass and edge parity, while position/yaw still has no passing learned checkpoint even when the best medium candidate has edge parity.

Readiness promotion command:

```bash
python scripts/build_sixdof_native_parity.py \
  --task position_yaw \
  --reset-profile position_yaw_easy \
  --reset-profile position_yaw_medium \
  --reset-profile broad \
  --num-envs 256 \
  --steps 200 \
  --seed 333 \
  --action-source teacher \
  --output artifacts/replay/sixdof_native_parity_current.json

python scripts/build_sixdof_readiness_report.py \
  --matrix artifacts/replay/sixdof_candidate_matrix_current.json \
  --room-report artifacts/replay/room_scan_autonomous_35s.room.json \
  --native-parity artifacts/replay/sixdof_native_parity_current.json \
  --output artifacts/replay/sixdof_readiness_current_native_parity.json
```

| scope | selected label | ready | failures | latency us | completed | pos err m | clearance p01 m |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| obstacle_avoidance | obstacle_focus | true | none | 9.967 | 1.0000 | 0.1402 | 0.4839 |
| position_yaw | history1_h128 | false | sim_gate | 9.323 | 0.6602 | 3.8810 | 0.0889 |
| multitask | safe_horizon800 | false | sim_gate | 9.996 | 0.5547 | 2.3717 | 0.0613 |

Global evidence used by the readiness report: room map ready with 7416 points, and Python/native parity passed across easy, medium, and broad resets with worst state RMSE `2.96e-7`, worst ranger RMSE `0.00151` mm, and zero terminal mismatches. The readiness report now carries the matrix's best multi-task candidate alongside single-task candidates. After edge export, the best multi-task candidate is blocked only by the sim gate. Per-task gate reporting shows `safe_horizon800` still fails position_yaw and circle on clearance/completion/position error, while obstacle_avoidance is down to completion/position error only. This report is a simulation/edge-bench promotion gate, not approval for autonomous live flight.

Yaw-aware validation pass: checkpoint and suite evaluators now report `mean_yaw_error_rad` and `yaw_error_p95_rad`, and can gate with `--max-yaw-error-rad` plus `--max-yaw-p95-error-rad`.

```bash
python scripts/evaluate_sixdof_checkpoint.py --teacher --task position_yaw --reset-profile position_yaw_medium --steps 400 --num-envs 256 --native-step --max-yaw-error-rad 0.35 --max-yaw-p95-error-rad 0.60 --output artifacts/replay/sixdof_teacher_position_yaw_medium_yaw_gate.json
python scripts/evaluate_sixdof_checkpoint.py --checkpoint artifacts/curriculum/position_yaw/easy_medium_history1_h128/checkpoint.pt --task position_yaw --reset-profile position_yaw_medium --steps 400 --num-envs 256 --native-step --max-yaw-error-rad 0.35 --max-yaw-p95-error-rad 0.60 --output artifacts/replay/sixdof_history1_h128_medium_yaw_gate.json
```

| controller | passed | failures | mean yaw err rad | yaw p95 rad | completed | pos err m |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| teacher | true | none | 0.0011 | 0.3995 | 1.0000 | 0.1354 |
| history1_h128 | false | min_clearance, completion, position_error | 0.1306 | 0.4677 | 0.6250 | 3.6545 |

The current history checkpoint is not primarily yaw-limited on the medium profile; it still fails position/completion/clearance. The yaw metrics are now present so future position/orientation checkpoints cannot pass while hiding heading errors.

A shorter 200-step suite with the same yaw thresholds passed for `history1_h128` (`completed=0.9453`, `pos_err=0.8279m`, `mean_yaw_error=0.0284rad`), which confirms the checkpoint can hold briefly on the medium reset. It does not override the longer 400/800-step failures; use the short suite only as a smoke signal for future curriculum work.

Candidate matrix refresh: the matrix now ranks with yaw error when available, treats edge parity as pass/fail/missing instead of only present/missing, and surfaces the best multi-task checkpoint separately.

Best multi-task edge evidence command:

```bash
python scripts/build_sixdof_edge_evidence.py \
  --matrix artifacts/replay/sixdof_candidate_matrix_current.json \
  --include-best-multitask \
  --run \
  --report artifacts/replay/sixdof_edge_evidence_multitask.json
```

Result: `safe_horizon800` exported with TorchScript parity max error `0.0` and latency `9.996 us/sample`.

Training sweep note: task loss weights and one-iteration task-probability DAgger sampling both underperformed `safe_horizon800`; the next serious attempt should change rollout length/curriculum or use closed-loop PPO-style fine-tuning. Detailed commands and tables are in `docs/benchmarks/sixdof_training_sweeps_2026-05-20.md`.

```bash
python scripts/build_sixdof_candidate_matrix.py \
  --suite artifacts/replay/sixdof_candidate_suite_safe_tasks.json \
  --suite artifacts/replay/sixdof_obstacle_focus_strict_suite.json \
  --suite artifacts/replay/sixdof_position_yaw_ppo_progress_medium_suite.json \
  --suite artifacts/replay/sixdof_history1_medium_suite.json \
  --parity obstacle_focus=artifacts/edge/sixdof_obstacle_focus_refine.parity.json \
  --parity history1_h128=artifacts/edge/sixdof_history1_h128.parity.json \
  --parity safe_horizon800=artifacts/edge/sixdof_safe_horizon800.parity.json \
  --latency obstacle_focus=artifacts/edge/sixdof_obstacle_focus_refine.latency.json \
  --latency history1_h128=artifacts/edge/sixdof_history1_h128.latency.json \
  --latency safe_horizon800=artifacts/edge/sixdof_safe_horizon800.latency.json \
  --output artifacts/replay/sixdof_candidate_matrix_current.json
```

Current refreshed selections:

| category | selected label | passed | edge | completed | pos err m | clearance p01 m |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| obstacle_avoidance | obstacle_focus | true | pass | 1.0000 | 0.1402 | 0.4839 |
| position_yaw | history1_h128 | false | pass | 0.6602 | 3.8810 | 0.0889 |
| multitask | safe_horizon800 | false | pass | 0.5547 | 2.3717 | 0.0613 |

The refreshed matrix preserves the same deployment boundary: obstacle avoidance is the only learned checkpoint with both a passing sim gate and edge evidence. The strongest multi-task checkpoint now has edge evidence too, but it remains blocked by clearance, completion, and position-error failures.

Per-task gate refresh:

| checkpoint | position_yaw | obstacle_avoidance | circle |
| --- | --- | --- | --- |
| safe_horizon800 | min_clearance, completion, position_error | completion, position_error | min_clearance, completion, position_error |

Task-weighted offline smoke: offline and DAgger retraining now accept repeatable `--task-weight TASK=WEIGHT` flags so follow-up multi-task runs can emphasize tasks identified by the per-task gate report.

```bash
python scripts/train_sixdof_offline.py \
  --dataset artifacts/dagger/sixdof_safe_tasks_horizon800/iter_01.npz \
  --checkpoint artifacts/checkpoints/sixdof_safe_horizon800_task_weight_smoke.pt \
  --epochs 1 \
  --batch-size 8192 \
  --hidden-size 128 \
  --learning-rate 8e-4 \
  --eval-steps 30 \
  --eval-num-envs 32 \
  --select-by-eval \
  --native-step \
  --task-weight position_yaw=1.5 \
  --task-weight circle=1.5
```

The checkpoint stores `task_weights={"position_yaw": 1.5, "circle": 1.5}`. A 300-step native validation smoke did not pass (`completed=0.1953`, `pos_err=5.4099m`, `clearance_p01=0.0340m`), so this is a verified training knob, not a better candidate yet.

Task-weight sweep runner: added `scripts/run_sixdof_task_weight_sweep.py` to compare task-weight variants with a train command plus suite gate command per variant.

Dry-run manifest:

```bash
python scripts/run_sixdof_task_weight_sweep.py \
  --max-variants 2 \
  --baseline-checkpoint artifacts/dagger/sixdof_safe_tasks_horizon800/iter_01.pt \
  --report artifacts/replay/sixdof_task_weight_sweep_manifest.json \
  --output-dir artifacts/task_weight_sweep/manifest
```

Short execution smoke:

```bash
python scripts/run_sixdof_task_weight_sweep.py \
  --max-variants 1 \
  --run \
  --eval-steps 30 \
  --eval-num-envs 32 \
  --suite-steps 80 \
  --suite-num-envs 64 \
  --report artifacts/replay/sixdof_task_weight_sweep_smoke.json \
  --output-dir artifacts/task_weight_sweep/smoke
```

Result: `balanced_control` completed the short suite with `completed=1.0000`, `pos_err=0.9555m`, and `clearance_p01=1.1176m`. This confirms the sweep automation path; it is not a replacement for the current 300-step validation gate.

300-step task-weight sweep with baseline:

```bash
python scripts/run_sixdof_task_weight_sweep.py \
  --baseline-checkpoint artifacts/dagger/sixdof_safe_tasks_horizon800/iter_01.pt \
  --run \
  --eval-steps 80 \
  --eval-num-envs 64 \
  --suite-steps 300 \
  --suite-num-envs 128 \
  --report artifacts/replay/sixdof_task_weight_sweep_300.json \
  --output-dir artifacts/task_weight_sweep/suite300
```

| variant | completed | pos err m | clearance p01 m |
| --- | ---: | ---: | ---: |
| baseline | 0.5651 | 2.8909 | 0.0535 |
| balanced_control | 0.2526 | 4.9693 | 0.0390 |
| focus_position_circle_15 | 0.2552 | 5.1258 | 0.0388 |
| focus_position_circle_2 | 0.2370 | 5.2127 | 0.0365 |
| focus_circle_2 | 0.2005 | 5.2286 | 0.0413 |
| focus_position_2 | 0.2500 | 5.0548 | 0.0416 |
| focus_position_circle_h256 | 0.2552 | 4.8840 | 0.0429 |

Conclusion: short static-imitation retraining with task weights underperforms the existing `safe_horizon800` baseline. The next multi-task attempt should change rollout collection or closed-loop optimization, not only static sample weights.
