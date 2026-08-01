# Benchmark artifacts

Benchmark results are generated evidence, not maintained documentation. Write
new machine-readable reports below `artifacts/` and bind them to the source
commit, exact command/configuration, seed set, metric contract, and input
hashes. Do not copy a dated run narrative into this directory as if it were a
current baseline.

The pre-review benchmark notes were removed from the active tree because their
checkpoints, metrics, or implementation paths are no longer current. They
remain recoverable from Git commit `038ee15` and have no deployment authority.

Current benchmark entrypoints include:

- `scripts/benchmark_sixdof_native.py` for native/Python simulator throughput;
- `scripts/benchmark_mujoco_sixdof.py` for desktop MuJoCo throughput;
- `scripts/benchmark_sixdof_desktop_policy.py` for desktop policy latency;
- `scripts/evaluate_sixdof_checkpoint.py` for a fresh six-DoF teacher or
  simulation-checkpoint gate;
- `scripts/evaluate_puffer_fixed_door_teacher.py` for the privileged fixed-door
  teacher metric.

None of these commands grants AI Deck shadow or flight authority. Edge-v3
promotion requirements live in `docs/edge_navigation_v3.md`.
