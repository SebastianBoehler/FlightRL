# 6-DoF Validation Suite Baseline

Date: 2026-05-20

Command:

```bash
python scripts/evaluate_sixdof_suite.py \
  --teacher teacher_safe position_yaw,obstacle_avoidance,circle \
  --candidate obstacle_focus artifacts/dagger/sixdof_obstacle_focus_refine/iter_02.pt obstacle_avoidance \
  --candidate multitask artifacts/checkpoints/sixdof_multitask_h256.pt checkpoint \
  --steps 300 \
  --num-envs 128 \
  --native-step \
  --output artifacts/replay/sixdof_validation_suite_latest.json
```

Result:

| label | controller | tasks | passed | failures | pos err m | clearance p01 m | completed | teacher L2 |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| teacher_safe | teacher | position_yaw, obstacle_avoidance, circle | true | none | 0.4997 | 0.1939 | 0.9922 | 0.0000 |
| obstacle_focus | checkpoint | obstacle_avoidance | true | none | 0.3971 | 0.5148 | 1.0000 | 0.0164 |
| multitask | checkpoint | position_yaw, obstacle_avoidance, attitude, circle | false | min_clearance, completion, position_error | 7.7712 | 0.0283 | 0.1445 | 0.5797 |

Interpretation:

- The current analytic teacher reference remains viable for position/yaw, obstacle avoidance, and circle tasks under this shorter native-step suite.
- `artifacts/dagger/sixdof_obstacle_focus_refine/iter_02.pt` is still the strongest obstacle-specific learned checkpoint.
- `artifacts/checkpoints/sixdof_multitask_h256.pt` is not a deployment candidate. The attitude task and broad multi-task objective still need additional objective work or separate task-specific refinement.

This is simulation validation only. It does not approve live Crazyflie hardware deployment.
