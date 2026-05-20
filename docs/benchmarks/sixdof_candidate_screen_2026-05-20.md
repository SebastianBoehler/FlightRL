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

| task | selected label | ready | failures | latency us | completed | pos err m | clearance p01 m |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| obstacle_avoidance | obstacle_focus | true | none | 9.967 | 1.0000 | 0.1402 | 0.4839 |
| position_yaw | history1_h128 | false | sim_gate | 9.323 | 0.6602 | 3.8810 | 0.0889 |

Global evidence used by the readiness report: room map ready with 7416 points, and Python/native parity passed across easy, medium, and broad resets with worst state RMSE `2.96e-7`, worst ranger RMSE `0.00151` mm, and zero terminal mismatches. This report is a simulation/edge-bench promotion gate, not approval for autonomous live flight.

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

Current refreshed selections:

| category | selected label | passed | edge | completed | pos err m | clearance p01 m |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| obstacle_avoidance | obstacle_focus | true | pass | 1.0000 | 0.1402 | 0.4839 |
| position_yaw | history1_h128 | false | pass | 0.6602 | 3.8810 | 0.0889 |
| multitask | safe_horizon800 | false | missing | 0.5078 | 3.0538 | 0.0446 |

The refreshed matrix preserves the same deployment boundary: obstacle avoidance is the only learned checkpoint with a passing sim gate and edge evidence. The strongest multi-task checkpoint is now visible, but it remains blocked by clearance, completion, and position-error failures.
