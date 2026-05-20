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
