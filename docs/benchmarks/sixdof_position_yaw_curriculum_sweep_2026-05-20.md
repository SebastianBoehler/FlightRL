# 6-DoF Position/Yaw Curriculum Sweep Smoke

Date: 2026-05-20

Change: added `scripts/run_sixdof_curriculum_sweep.py` to plan and execute staged position/yaw curriculum sweeps. Each variant builds one or more teacher datasets, trains an offline checkpoint with eval-based selection, then writes medium and broad gate reports.

Command:

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

Conclusion: the sweep runner is working and the first h128 staged curriculum checkpoint is comparable to prior manual curriculum attempts, but it still fails position/yaw gates. Continue with the planned h256 and broad-included variants before promoting any position/orientation checkpoint.
