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
