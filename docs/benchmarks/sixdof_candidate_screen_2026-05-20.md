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
