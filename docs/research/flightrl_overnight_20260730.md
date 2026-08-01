# FlightRL overnight lab notebook

Window: 2026-07-30 22:59 to 2026-07-31 07:30 Europe/Berlin  
Scope: offline-only recurrent fixed-door student work. No drone, radio, arming,
flight, flashing, or live authority.

## Frozen baseline

- Candidate: `artifacts/puffer_fixed_door_d1_v59_fresh_control_bc1m/flightrl_fixed_door_d1_seed11_1048576.bin`
- Checkpoint SHA-256: `f676d12b9d37c27f4cc62f99beceec8f30e74c88be8564cb242c23755e202cce`
- Corrected evaluation: `artifacts/puffer_fixed_door_d1_v59_fresh_control_bc1m/flightrl_fixed_door_d1_seed11_1048576.reevaluation.json`
- Evaluation SHA-256 after metadata-only legacy-contract annotation:
  `b919e4f9951ad28904ce6cc7ee9b7a0f7b76ee70fba387e673bcda27a9bdcbbc`
  (the checkpoint and metric values are unchanged; the original evaluation
  file SHA was `f897916f...5254b8`; the final metadata annotation binds both
  the explicit v59 legacy action contract and recurrent policy contract).
- Parent perception checkpoint: `artifacts/puffer_fixed_door_d1_v53_asymmetric_conservative262k/flightrl_fixed_door_d1_seed11_262144.bin`
- Parent SHA-256: `0e831420c7d1d1a1a46979aa1e096122cf511bb8dfd51cd12510fbb94d815761`
- v59 train seed/evaluation seed: `11` / `10011`
- v59 budget: 128 agents x 64 steps x 128 BC updates =
  1,048,576 transitions; no PPO rollouts; obstacles disabled; layout diversity
  enabled for evaluation.
- v59 corrected metrics over 359 full-camera episodes: 79.11% success,
  74.73% outside-FOV success, 0.56% collision; masked-camera success 1.53%.
- Repository checkpoint: `3cc77427a643aa2509c348bf9ffbceaff23c4fd5`
  on `main`. The worktree began with 35 tracked modifications and 146 untracked
  files from earlier work; none are to be reverted.
- PufferLib base: detached `83e9fae5dcf33461506aa3b6671393331dd9c78e`
  with existing local generated assets and a modified `pufferlib/torch_pufferl.py`.

## Chronology

### 22:59-23:06 — Handoff reconstruction and baseline verification

- Read the thread retrospective, experiment-control protocol, and continuation
  handoff before editing.
- Verified v59, its parent, and the corrected reevaluation hashes.
- Confirmed there was no visible active FlightRL/Puffer job. Process visibility
  is sandbox-limited, but no open v59 artifact or recent training output was
  visible.
- Baseline focused verification:
  `.venv/bin/pytest -q tests/test_puffer4_door_control.py tests/test_puffer4_door_readiness.py tests/test_puffer4_door_shadow_io.py tests/test_puffer4_door_runtime.py tests/test_puffer4_door_training.py tests/test_semantic_yaw_authority.py`
  completed in 1.10 s: 30 passed.
- Non-actuating v59 dry runs completed. The fixed-door control dry run proposed
  normalized search yaw 0.2972, or 68.11 deg/s under the actual v59 training
  scale; no readiness artifact exists and no command was sent.
- Static defect reproduced: the active detector teacher emits dimensionless
  centroid/search actions, and the door step sends normalized yaw directly to
  native physics at 4 rad/s (229.18 deg/s), ignoring the declared 70 deg/s
  action ceiling. The unused geometry teacher does normalize against 70
  deg/s. v59's host adapter correctly preserves the executed legacy semantics,
  then the live envelope clamps to 8 deg/s.
- Lineage gaps in the v59 report: no exact command, elapsed time,
  observation/action contract hashes, scene-distribution identifiers, or
  Puffer working-tree hash.

Verdict: baseline evidence is intact, but neither v59 nor a retrained policy may
be promoted on the original report alone. Complete the Gate A audit, repair the
native yaw contract, and make later reports self-identifying.

### 23:07-23:22 — A0 contract mechanism repairs

#### A0-YAW

- Hypothesis: mapping the normalized door yaw action through the declared
  70 deg/s ceiling, while feeding back the actually executed action in that
  declared normalization, removes the 229.18 deg/s native mismatch without
  changing labels, reward action cost, observation layout, or live limits.
- Parent checkpoint: v59 SHA `f676d12b...202cce`; no checkpoint was modified.
- One controlled variable: native yaw execution changes from direct normalized
  `action * 4 rad/s` to
  `action * radians(70 deg/s)`, with saturation round-tripped into the
  previous-action observation.
- Command: `.venv/bin/pytest -q tests/test_puffer4_door_action.py`.
- Runtime/seeds: 0.63 s; deterministic C mechanism test, no stochastic seed.
- Artifacts: `src/flightrl/native/native_door_action.{c,h}`,
  `tests/test_puffer4_door_action.py`; the native binding was split below the
  300-LOC soft limit using `native_door_domain.inc`.
- Metrics: normalized yaw `-0.5` maps to low-level `-0.1527163` under a
  4 rad/s physics ceiling and round-trips as previous action `-0.5`.
- Verdict: mechanism pass. This does not validate mission quality; every
  corrected-yaw candidate must be retrained and evaluated from scratch.
- Next decision: rebuild the exact native environment, re-run the teacher
  upper-bound gate, then run matched fresh-controller BC and DAgger.

#### A0-HOST integration suite: four isolated mechanism repairs

This was not treated as one causal policy experiment. Four independent
red-green mechanism tests each changed one boundary behavior while holding the
checkpoint and every other boundary fixed:

1. altitude-parity hypothesis: replacing the arbitrary host `z/3.0` with the
   native base-room `z/2.5` is sufficient to align that sensor field;
2. mask-parity hypothesis: quantize/resize before masking and fill from the
   quantized global mean to align the native mask statistic;
3. telemetry-completeness hypothesis: request `vx/vy/vz` so the recurrent
   host observation does not silently substitute zero velocities;
4. evidence-causality hypothesis: carry source-frame age into detector
   evidence so an already stale frame cannot refresh yaw authority.

- Parent for each: unchanged v59 runtime/checkpoint.
- Commands:
  `.venv/bin/pytest -q tests/test_puffer4_door_observation.py
  tests/test_puffer4_door_runtime.py tests/test_puffer4_door_self_mask.py
  tests/test_semantic_yaw_authority.py` and the broader focused Gate A set.
- Runtime/seeds: 0.54-0.55 s; deterministic fixtures, no stochastic seed.
- Artifacts: the corresponding observation runtime, self-mask, telemetry
  logger, grounding-age/yaw-authority modules and their focused tests only; no
  hardware output.
- Metrics: each new red fixture failed for its intended single reason; after
  the four separate repairs, 14/14 narrow parity tests and the 23/23 combined
  Gate A integration slice passed.
- Verdict: all four mechanisms pass independently, followed by integration
  pass. No mission-quality conclusion is attributed to the combined suite.
  Physical yaw authority remains disabled because no matching real shadow
  exists.
- Next decision for all four: strengthen report/shadow hash binding, then
  proceed to the offline native build and registered T60 comparison.

### 23:22-23:34 — A0 single-source-of-truth action contract

- Hypothesis: an immutable unit-bearing contract, rather than independent
  literals in native export, evaluation, host inference, and live safety code,
  will make an ignored or reinterpreted yaw setting fail closed.
- Parent checkpoint: unchanged v59 SHA `f676d12b...202cce`; no weights or
  recorded metrics changed.
- One controlled variable: action-scale configuration was centralized. The
  corrected contract is `fixed-door-declared-yaw-v1` (0.55 m/s, 70 deg/s,
  4 rad/s physics ceiling); v59 is explicitly
  `fixed-door-v59-legacy-physics-yaw-v1` (229.183 deg/s); the 8 deg/s yaw-only
  live envelope remains a separate safety contract.
- Commands:
  `<python-3.13> -m pytest -q
  tests/test_puffer4_door_contract.py tests/test_puffer4_door_control.py
  tests/test_puffer4_door_checkpoint.py tests/test_puffer4_door_action.py
  tests/test_puffer4_door_readiness.py tests/test_puffer4_door_export.py`,
  followed by `py_compile` on both trainers and the evaluator.
- Runtime/seeds: 0.97 s for 15 tests; deterministic contract fixtures, no
  stochastic seed.
- Artifacts: `src/flightrl/puffer4_door_contract.py` and its focused tests;
  generated native environment values, host scaling, evaluation reports, and
  readiness limits now consume the contract. Both v59 JSON reports were
  annotated with the hash-verified legacy contract.
- Metrics: red tests rejected the absent contract applicator and identifier;
  green result 15/15. Contract mutation and a runtime config value of
  229.183 deg/s under the corrected contract both raise errors.
- Verdict: mechanism pass. Generated config plus runtime verification now
  prevent the original silent bypass; legacy evidence remains readable only
  through an explicit legacy declaration.
- Next decision: add exact runtime/provenance to new reports, rebuild the native
  environment, and launch the matched seed-11 BC/DAgger pair.

### 23:34-23:45 — World Action Model feasibility side audit

- Hypothesis: a published WAM might offer a smaller or more causal way to
  improve outside-FOV recovery than the registered BC/DAgger comparison.
- Parent checkpoint: v59; no weights, environment, or experiment variable
  changed.
- One controlled variable: none; primary-source literature audit only.
- Command/runtime/seeds: web/paper review, approximately 11 minutes; no
  stochastic seed.
- Artifact: `docs/research/flightrl_world_action_models_20260730.md`.
- Metrics: current relevant WAMs are generally 2B-14B parameters and roughly
  0.5-7 Hz on datacenter GPUs. The UAV-specific WorldFly is simulation-only
  and reports 31% success on unseen intersections with coarse 3-9 m and
  30-degree action primitives. These are not comparable to the 80,997-
  parameter, bounded setpoint FlightRL lane.
- Verdict: do not introduce a WAM tonight or into the first live gate. A later
  causal screen may add one training-only action-conditioned next-latent head,
  remove it at inference, and otherwise keep the actor, streams, budget, and
  seeds fixed.
- Next decision: finish yaw-contract BC/DAgger evidence first. WAM work remains
  a separate post-gate representation experiment.

### 23:36-23:54 — T60-YAW seed-11 pilot

- Hypothesis: retraining the fresh recurrent controller under the corrected
  70 deg/s execution contract will preserve or improve v59 mission behavior
  while reducing the policy-to-live yaw-scale gap.
- Parent checkpoint: v53 perception tensors, SHA
  `0e831420...815761`; fresh fusion, MinGRU recurrence, and decoder initialized
  deterministically with seed 11.
- One controlled policy variable: native yaw execution changed from v59's
  legacy 4 rad/s direct ceiling to the declared 70 deg/s mapping. No policy
  roll-in, PPO, obstacles, camera randomization, or extra room intervention.
- Command:
  `<python-3.13>
  scripts/train_puffer_fixed_door_asymmetric.py --puffer-root
  <pufferlib-root>
  --source-checkpoint artifacts/puffer_fixed_door_d1_v53_asymmetric_conservative262k/flightrl_fixed_door_d1_seed11_262144.bin
  --source-report artifacts/puffer_fixed_door_d1_v53_asymmetric_conservative262k/flightrl_fixed_door_d1_seed11_262144.report.json
  --fresh-control --agents 128 --horizon 64 --bootstrap-updates 128
  --bootstrap-learning-rate 0.001 --rollouts 0 --screen-steps 1400
  --eval-steps 11000 --output-dir
  artifacts/puffer_fixed_door_d1_t60_yaw_bc_seed11
  --bootstrap-max-policy-rollin 0 --seed 11 --evaluation-seed 10011`.
- Runtime/seeds: 1,007.87 s wall time; train 11, held-out evaluation 10011,
  appearance base 2003. Native Puffer environment was rebuilt.
- Artifacts:
  `artifacts/puffer_fixed_door_d1_t60_yaw_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.bin`,
  SHA `a2a1a8fa...c1066d`; report SHA `114fb298...2edbf`.
  Parent, action-contract, command, interpreter, source, generated config,
  native extension, and Puffer diff hashes are embedded in the report.
- Metrics: selected `bootstrap`; full-camera 86.23% success over 1,917
  episodes, 74.01% outside-FOV success, 0.68% collision; masked-camera 0.00%
  over 1,024 episodes. Teacher upper bound was 98.61% success, 97.16%
  outside-FOV, and 0.22% collision. All aggregate simulation gates passed.
- Verdict: promising diagnostic (+7.12 completion points over v59), not
  promotable. Appearance RNG is consumed per control step, so later scene
  appearances can diverge between policies with different episode lengths;
  equal seeds do not yet prove a matched BC/DAgger stream. Temporal/group and
  live-cap evidence are also pending.
- Next decision: freeze this pilot, repair episode-indexed physical/appearance
  streams, then rerun both seed-11 BC and DAgger from the identical repaired
  native build. Do not compare this pilot to a post-repair DAgger run.

### 23:54-00:06 — A0 observation/report/readiness contract hardening

#### A0-OBS signed temporal channel

- Hypothesis: host and native temporal preprocessing are not fully equivalent
  for a frame that becomes darker.
- Parent checkpoint: v59; no policy weights changed.
- One controlled variable: host delta arithmetic casts quantized `uint8`
  frames to float before subtraction.
- Command:
  `<python-3.13> -m pytest -q
  tests/test_puffer4_door_runtime.py tests/test_puffer4_door_observation.py
  tests/test_native_sixdof_vision.py tests/test_puffer4_door_control.py`.
- Runtime/seeds/artifacts: 0.52 s; deterministic frames; runtime and focused
  tests only.
- Metrics: the red fixture showed a native-expected `-17/255` delta becoming
  `+239/255` through NumPy unsigned underflow; after the cast, 17/17 tests pass
  and the motion bit remains zero because `17/255 < 0.08`.
- Verdict: transfer-blocking host bug fixed. The full 9,248-float contract is
  now hash-bound as `fixed-door-recurrent-policy-v1`: 9,242 deployable values,
  6 privileged training-only values, exact channel/segment ordering,
  signed-delta preprocessing, 96-unit single-layer MinGRU, terminal reset, and
  executed previous-action semantics.
- Next decision: require both action and policy contract hashes in reports,
  shadow summaries, the host adapter, and readiness.

#### A0-GATE fail-closed morning evidence

- Hypothesis: row-count and copied-summary checks can approve a short,
  geometrically wrong, stale, or contract-mismatched shadow trace.
- Parent: v59 metadata only; checkpoint SHA remains
  `f676d12b...202cce`.
- One controlled variable: readiness evidence validation changed from trusting
  summary fields to recomputing and hash-binding the supplied CSV/report.
- Commands: focused readiness/shadow/contract tests plus a non-actuating v59
  `--dry-run`; no Crazyflie or radio connection was made.
- Runtime/seeds/artifacts: 0.51-0.58 s test runs; deterministic fixtures; no
  stochastic seed. `puffer4_door_readiness.py` and
  `puffer4_door_shadow_io.py` remain below 300 LOC.
- Metrics: gate requires effective sampled coverage at least 20 s (timestamp
  span plus one median frame interval), strictly increasing timestamps,
  constant near-4:3 frames at least 128x96, search and target/recovery phase
  coverage, finite non-actuating rows, detection/sign samples, dropped-frame,
  inference, and grounding-age limits. It binds checkpoint, report, CSV,
  action, recurrent-policy, and live-safety hashes. Strict off-center zero-yaw
  remains incorrect. The focused combined suite passed 33/33 before the final
  policy binding and 19/19 afterward.
- Verdict: software gate hardened; every old summary fails closed until it is
  regenerated against a matching real CSV. The known v53 trace has healthy
  duration/geometry but correctly fails phase coverage and cannot approve
  v59.
- Next decision: the first operator command remains a 20 s non-actuating
  shadow. No live authority exists until its newly generated evidence passes
  and is manually reviewed.

### 23:55-00:06 — A0-RNG matched procedural streams and group logging

- Hypothesis: deriving both native random streams from
  `(base seed, appearance seed, environment index, episode index)` at every
  reset makes the nth BC and DAgger episode reproducible regardless of the
  preceding policy's episode length.
- Parent checkpoint: none; experiment-control infrastructure only. The T60
  seed-11 pilot remains frozen and cannot be compared to post-repair runs.
- One controlled variable: rolling cross-episode physical/appearance RNG was
  replaced by domain-separated episode-indexed seeds. Scene distributions,
  observation/action contracts, teacher, reward, and dynamics are unchanged,
  although exact procedural identities deliberately change.
- Commands: focused ctypes C/export red-green tests; all
  `tests/test_puffer4_door_*.py`; Ruff; C11 compilation with
  `-Wall -Wextra -Werror`; native CPU Puffer rebuild; real vector reset,
  multi-environment, and log-clearing smokes.
- Runtime/seeds: deterministic golden-vector tests plus Puffer vector smoke;
  training/evaluation seeds were not consumed for a policy run.
- Artifacts: `native_door_episode_rng.{c,h}`,
  `native_door_episode_groups.inc`, native env integration/export, and focused
  tests. Rebuilt Puffer extension SHA
  `0c31af71...bb74a`.
- Metrics: red state was 1 failure plus 5 missing-helper errors; final focused
  7/7 and broad 80/80 tests passed. Native smokes showed bit-identical next
  resets after 1 versus 7 preceding steps, four distinct/reproducible
  environment streams, and correct aggregation/clearing of 28 log metrics.
- Worst-group instrumentation: schema 1 logs marginal layout family, door
  face, low-light, obstacle, and outside-FOV supports/successes without
  consuming RNG. Category zero is derived from totals. These are marginal,
  not intersectional, worst groups.
- Verdict: mechanism and native integration pass. The source/build lane is
  frozen for a new matched pair.
- Next decision: rerun corrected-yaw seed-11 BC and DAgger from this exact
  build; confirm their deterministic `candidate-source` tensors match before
  interpreting the treatment delta.

### 00:11-00:28 — T61-BC corrected-yaw episode-indexed baseline, seed 11

- Hypothesis: corrected 70 deg/s native yaw plus fresh-controller BC remains
  stronger than v59 after repairing the episode stream, and establishes the
  exact paired baseline for a DAgger distribution-shift test.
- Parent checkpoint: v53 perception tensors only,
  `0e831420...d815761`; all control/fusion/recurrent weights were freshly
  initialized. The authoritative comparison baseline remains v59,
  `f676d12b...202cce`.
- One controlled variable: this arm uses pure expert BC with maximum policy
  roll-in `0.0`. Corrected action, observation/recurrence, teacher, dynamics,
  budget, and episode-indexed procedural stream are frozen for its paired
  DAgger arm.
- Command:
  `<python-3.13>
  scripts/train_puffer_fixed_door_asymmetric.py --puffer-root
  <pufferlib-root>
  --source-checkpoint
  artifacts/puffer_fixed_door_d1_v53_asymmetric_conservative262k/flightrl_fixed_door_d1_seed11_262144.bin
  --source-report
  artifacts/puffer_fixed_door_d1_v53_asymmetric_conservative262k/flightrl_fixed_door_d1_seed11_262144.report.json
  --fresh-control --agents 128 --horizon 64 --bootstrap-updates 128
  --bootstrap-learning-rate 0.001 --rollouts 0 --screen-steps 1400
  --eval-steps 11000 --output-dir
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11
  --bootstrap-max-policy-rollin 0 --seed 11 --evaluation-seed 10011
  --skip-build`.
- Runtime/seeds: 1,011.86 s wall time; train seed 11, held-out evaluation
  seed 10011, episode-indexed physical and appearance streams.
- Artifacts: selected bootstrap checkpoint
  `artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.bin`,
  SHA `364647c6...8946`; report SHA `0151023e...457e`;
  deterministic source candidate SHA `844980c3...82d3`. Report-bound contracts
  are corrected action `55fc561c...e27c`, recurrent policy
  `ad6fa58f...8058`, and episode stream `9d2db780...c4b`.
- Metrics: full-camera 89.64% success over 2,028 episodes, 79.78%
  outside-FOV success, 0.30% collision; masked-camera 0.00% over 1,024
  episodes. Teacher upper bound was 98.92% success, 98.26% outside-FOV, and
  0.39% collision. Worst supported marginal was layout family 2 at 88.48%
  success; derived layout family 0 was 88.58%, and the worst door-face
  marginal was derived face 0 at 88.76%. Low-light and obstacle supports were
  zero and therefore are not robustness evidence.
- Verdict: passes the aggregate simulation screen and exceeds v59 by 10.53
  completion points with lower collision. It is not yet promoted: seed 11 is
  the only seed, temporal-order/live-cap/robustness evaluations are pending,
  and the paired DAgger comparison has not finished.
- Next decision: run the matched DAgger seed-11 arm with policy roll-in `0.5`
  as the sole intervention; require identical source candidate and contract
  hashes before attributing any metric delta.

### 00:30-00:51 — T61-DAGGER matched policy-roll-in treatment, seed 11

- Hypothesis: corrective on-policy aggregation will reduce fresh-controller
  distribution shift and improve completion by at least five points over the
  repaired, corrected-yaw BC arm.
- Parent checkpoint: the same v53 perception-only parent as T61-BC. Both arms
  have exact source candidate SHA `844980c3...82d3`.
- One controlled variable: maximum policy roll-in increased from BC's `0.0`
  to `0.5`. Seed, parent, initial control weights, 1,048,576-transition
  budget, teacher, optimizer, action/policy/stream contracts, and held-out
  evaluation stream are identical.
- Command:
  `<python-3.13>
  scripts/train_puffer_fixed_door_asymmetric.py --puffer-root
  <pufferlib-root>
  --source-checkpoint
  artifacts/puffer_fixed_door_d1_v53_asymmetric_conservative262k/flightrl_fixed_door_d1_seed11_262144.bin
  --source-report
  artifacts/puffer_fixed_door_d1_v53_asymmetric_conservative262k/flightrl_fixed_door_d1_seed11_262144.report.json
  --fresh-control --agents 128 --horizon 64 --bootstrap-updates 128
  --bootstrap-learning-rate 0.001 --rollouts 0 --screen-steps 1400
  --eval-steps 11000 --output-dir
  artifacts/puffer_fixed_door_d1_t61_episode_rng_dagger50_seed11
  --bootstrap-max-policy-rollin 0.5 --seed 11 --evaluation-seed 10011
  --skip-build`.
- Runtime/seeds: 1,202.72 s wall time; train seed 11 and evaluation seed
  10011.
- Artifacts: selected bootstrap checkpoint
  `artifacts/puffer_fixed_door_d1_t61_episode_rng_dagger50_seed11/flightrl_fixed_door_d1_seed11_1048576.bin`,
  SHA
  `ed516887acd165a8a7613e3ae562e21bc05cb1d587d6aa395afe99a3fdd68cef`;
  report SHA
  `dbdc05efaf1d168216fea6596af557712096b78a44f9cba55b526784ea76cac2`;
  source candidate SHA
  `844980c3...82d3`.
- Metrics: full-camera 87.99% success over 1,999 episodes, 76.96%
  outside-FOV success, 0.40% collision; masked-camera 0.00% over 1,024
  episodes. The teacher and scene stream reproduce the BC arm exactly. The
  worst supported/derived marginal was layout family 0 at 85.37%.
- Verdict: treatment rejected. Relative to matched BC, DAgger changes mission
  success by -1.65 points, outside-FOV success by -2.82 points, and collision
  by +0.10 points. It does not explain or correct the residual failures and
  fails the preregistered gain/kill criterion.
- Next decision: stop generating DAgger versions. Validate the stronger pure-BC
  arm on seeds 23 and 47, then spend remaining budget on promotion ablations
  and separately controlled robustness tests.

### 00:51-01:08 — T61-BC validation, seed 23

- Hypothesis: corrected-yaw fresh-controller BC retains a meaningful gain over
  v59 under a second initialization and disjoint procedural/evaluation stream.
- Parent checkpoint: the same v53 perception-only parent and exact frozen
  source/action/policy/episode-stream contracts as seed 11.
- One controlled variable: train seed changes from 11 to 23 and its held-out
  evaluation stream changes from 10011 to 10023. Algorithm, hyperparameters,
  budget, parent, and environment distribution are unchanged.
- Command:
  `<python-3.13>
  scripts/train_puffer_fixed_door_asymmetric.py --puffer-root
  <pufferlib-root>
  --source-checkpoint
  artifacts/puffer_fixed_door_d1_v53_asymmetric_conservative262k/flightrl_fixed_door_d1_seed11_262144.bin
  --source-report
  artifacts/puffer_fixed_door_d1_v53_asymmetric_conservative262k/flightrl_fixed_door_d1_seed11_262144.report.json
  --fresh-control --agents 128 --horizon 64 --bootstrap-updates 128
  --bootstrap-learning-rate 0.001 --rollouts 0 --screen-steps 1400
  --eval-steps 11000 --output-dir
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed23
  --bootstrap-max-policy-rollin 0 --seed 23 --evaluation-seed 10023
  --skip-build`.
- Runtime/seeds: 987.65 s wall time; train 23, evaluation 10023.
- Artifacts: selected bootstrap checkpoint
  `artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed23/flightrl_fixed_door_d1_seed23_1048576.bin`,
  SHA
  `cfaa2e9296aee15b406b78f8bf5cc72ea19b546b6547094d70c6deb40d03e343`;
  report SHA
  `353d8161563681f84b502d723b6803bdc0986fb8f5ae9ed762ae0ebfe2b7b625`.
- Metrics: full-camera 87.03% success over 1,904 episodes, 74.54%
  outside-FOV success, 0.37% collision; masked-camera 0.00% success and 0.68%
  collision over 1,024 episodes. Worst supported marginal was layout family 1
  at 83.68%. Teacher was 98.71% success, 97.62% outside-FOV, and 0.34%
  collision.
- Verdict: candidate-only screen pass. This seed is 2.61 points below seed 11
  and is descriptively 7.92 completion points above the single authoritative
  v59 result, with comparable outside-FOV behavior and low collision. There is
  no same-seed matched v59 evaluation for seed 23; the camera-causality screen
  itself passes.
- Next decision: run seed 47 unchanged. Treat these in-training evaluations as
  screening evidence because candidate selection and final evaluation reuse a
  seed; use a new shared seed for the v59-versus-candidate promotion bridge.

### 01:08-01:25 — T61-BC validation, seed 47

- Hypothesis: the corrected-yaw BC gain survives a third controller
  initialization and procedural/evaluation stream.
- Parent checkpoint: unchanged v53 perception-only parent with the same frozen
  source and contract hashes as seeds 11 and 23.
- One controlled variable: train/evaluation seeds change to 47/10047. All
  algorithm, environment, budget, parent, and evaluation settings are fixed.
- Command:
  `<python-3.13>
  scripts/train_puffer_fixed_door_asymmetric.py --puffer-root
  <pufferlib-root>
  --source-checkpoint
  artifacts/puffer_fixed_door_d1_v53_asymmetric_conservative262k/flightrl_fixed_door_d1_seed11_262144.bin
  --source-report
  artifacts/puffer_fixed_door_d1_v53_asymmetric_conservative262k/flightrl_fixed_door_d1_seed11_262144.report.json
  --fresh-control --agents 128 --horizon 64 --bootstrap-updates 128
  --bootstrap-learning-rate 0.001 --rollouts 0 --screen-steps 1400
  --eval-steps 11000 --output-dir
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed47
  --bootstrap-max-policy-rollin 0 --seed 47 --evaluation-seed 10047
  --skip-build`.
- Runtime/seeds: 982.62 s wall time; train 47, evaluation 10047.
- Artifacts: selected bootstrap checkpoint
  `artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed47/flightrl_fixed_door_d1_seed47_1048576.bin`,
  SHA
  `cc5ef33fb20fb04489e4a4f92c9b5e8329f8df63989d2f3181855da4f5802eb1`;
  report SHA
  `7de516e9bb19b4f244847c4810f5212cb446c5d620b1b6629a0f1ef5e0c4206e`.
- Metrics: full-camera 89.93% success over 2,055 episodes, 80.08%
  outside-FOV success, 0.49% collision; masked-camera 0.00% success and 0.00%
  collision over 1,024 episodes. Worst supported marginal was layout family 1
  at 86.71%. Teacher was 98.90% success, 98.20% outside-FOV, and 0.38%
  collision.
- Verdict: candidate-only screen pass. All three corrected-yaw BC screens are
  descriptively more than five completion points above the single v59 result,
  but seeds 23/47 do not have matched v59 evaluations. Candidate arithmetic
  means are 88.87% mission success, 78.13% outside-FOV success, 0.38%
  collision, and 0.00% masked-camera success. Per-seed mission range is
  87.03-89.93%; the worst observed seed/group combination is seed-23 layout
  family 1 at 83.68%.
- Next decision: retain seed 11 as the preregistered primary checkpoint rather
  than selecting the best validation seed. Harden lineage/build fail-closed
  checks, then compare it with v59 on a new shared held-out promotion seed.

### 01:25-02:05 — A0 typed contract and provenance enforcement

- Hypothesis: the v59 yaw defect becomes structurally difficult to repeat if
  policy semantics, simulator authority, and live safety each have one typed,
  immutable owner and every boundary rejects unknown, mutated, or mismatched
  contract bytes.
- Parent checkpoint: unchanged v59 and T61 seed-11 bytes; this was
  infrastructure-only work.
- One controlled variable: independent literals and report trust were replaced
  by a versioned contract/provenance graph. The three deliberately different
  quantities remain separate: corrected policy yaw is 70 deg/s, the native
  physics ceiling is 4 rad/s, and the live yaw-only safety cap is 8 deg/s.
- Commands: focused red-green suites for action/policy/evidence-age contracts,
  checkpoint bundles, native build provenance, canonical/challenge evaluation,
  and report selection; then Ruff, `py_compile`, strict C11 warning builds, and
  line-count checks. The final independent evaluator/selection slice was
  `.venv/bin/pytest -q tests/test_puffer4_door_bundle.py
  tests/test_puffer4_door_selection.py tests/test_puffer4_door_runner.py
  tests/test_puffer4_door_provenance.py
  tests/test_puffer4_door_canonical_evaluation.py
  tests/test_puffer4_door_challenge_evaluation.py
  tests/test_puffer4_door_challenge_output.py
  tests/test_puffer4_door_challenge_runner.py
  tests/test_puffer4_door_challenge_specs.py
  tests/test_evaluate_puffer_fixed_door_checkpoint_cli.py`.
- Runtime/seeds: 2.31 s for the final 56-test slice; deterministic fixtures,
  no policy seed.
- Artifacts: approved action registry, exact 9,248-float recurrent observation
  contract, episode-stream contract, runtime evidence-age contract, nested
  checkpoint lineage validator, and ABI/source/binary-bound native build
  fingerprint. The corrected action, policy, stream, and evidence-age contract
  SHAs are respectively `55fc561c...e27c`, `ad6fa58f...8058`,
  `9d2db780...c4b`, and `5223d412...f48`.
- Metrics: a self-consistent but re-hashed 71 deg/s contract, an approved
  contract borrowed from another lineage, a stale native binary, a preloaded
  wrong Puffer module, or a changed evaluation source manifest now fails
  closed. All source modules remain at or below the 300-LOC soft limit.
- Verdict: mechanism pass. “Single source of truth” here is not one global yaw
  number; it is one owner per physical meaning plus verified conversion edges.
  v59 remains loadable only through its exact legacy 229.183 deg/s declaration.
- Next decision: mint a fresh native build fingerprint and evaluate v59 and
  T61 seed 11 under the same new held-out stream and build.

### 02:05-02:22 — A0 preregistered checkpoint selector

- Hypothesis: a machine-enforced comparator prevents choosing the most
  attractive seed or silently comparing different budgets/builds/contracts.
- Parent checkpoint: T61 seed 11 remains the preregistered candidate; v59
  remains the authoritative baseline.
- One controlled variable: promotion judgment moved from informal notebook
  comparison to a fail-closed selection report. No policy, simulation, or
  threshold changed.
- Command: the 56-test evaluator/selection slice above plus
  `.venv/bin/python scripts/select_puffer_fixed_door_checkpoint.py --help`.
- Runtime/seeds: 2.31 s test slice; loading the real seed 11/23/47 training
  screens was deterministic and consumed no environment seed.
- Artifacts: `src/flightrl/puffer4_door_selection*.py`,
  `scripts/select_puffer_fixed_door_checkpoint.py`, and focused tests.
- Metrics: the selector requires matched evaluation seed, steps, agents,
  native fingerprint, procedural/evidence-age contracts, finite complete
  conditions, exact checkpoint lineage, and three candidate-seed training
  screens. Promotion requires at least +5 mission points, collision at most
  3%, no material outside-FOV/masked-camera/worst-group regression, and bounded
  latency/throughput regression. The 8 deg/s simulation screen is reported
  separately and the next physical gate is always `shadow_only`.
- Verdict: mechanism pass; no candidate is selected until the new held-out
  reports exist.
- Next decision: complete the adversarial shadow/control evidence review before
  freezing source and beginning the held-out evaluations.

### 02:22-02:43 — A0 shadow identity and live-gate adversarial hardening

- Hypothesis: a trace is not trustworthy unless detector results are causal,
  stream health is recomputed from rows, and inference consumes the same bytes
  that were hashed.
- Parent checkpoint: unchanged v59/T61 artifacts; no policy or simulation
  variable changed.
- One controlled variable: shadow/readiness evidence validation changed from
  path/summary trust to immutable byte snapshots plus causal row-derived
  evidence.
- Commands: focused TDD/adversarial review suites followed by the central
  `.venv/bin/pytest -q tests/test_puffer4_door*.py
  tests/test_evaluate_puffer_fixed_door_checkpoint_cli.py`; Ruff,
  `py_compile`, and an exact v59 shadow `--dry-run` to `/tmp`.
- Runtime/seeds: 6.24 s for the central 221-test slice; deterministic
  fixtures and one synthetic dry-run row, no stochastic environment seed.
- Artifacts: detector/run identity, immutable checkpoint/config snapshots,
  causal capture, row-derived detector/cadence/drop metrics, projection
  validator, and split runtime-policy helper. No real trace was created.
- Metrics: 221/221 central fixed-door tests passed. The dry run bound v59 SHA
  `f676d12b...202cce`, evaluation SHA `b919e4f...dcbbc`, legacy action,
  recurrent policy, runtime evidence-age, detector, hardware config, and
  monitor-only identity; forward and executed previous action were exactly
  zero and projected yaw was capped at 8 deg/s.
- Verdict: mechanism pass. Strict off-center detection with zero yaw remains
  incorrect. Remaining morning blockers are unpinned detector weight
  revisions, live height/duration bounds, output overwrite protection, and the
  possibility that an exact 20 s request samples slightly less than 20 s; all
  fail closed rather than enabling authority.
- Next decision: perform only a non-actuating real shadow in the morning and
  rerun to a new path if sampled coverage is short.

### 02:43-03:10 — P62 held-out promotion evaluation, T61 BC seed 11

- Hypothesis: the preregistered T61 seed-11 corrected-yaw BC checkpoint beats
  v59 by at least five mission points on a new shared stream while retaining
  collision, camera causality, worst marginals, recurrence, and runtime
  behavior.
- Parent checkpoint: v53 perception-only parent SHA
  `0e831420...815761`; evaluated checkpoint T61 seed 11 SHA
  `364647c6...8946`.
- One controlled variable: checkpoint identity. Evaluation environment,
  episode-indexed stream, seed 20011, 128 agents, 12,000 steps per condition,
  and the fingerprinted native CPU build were frozen for the upcoming v59
  comparator.
- Command:
  `<python-3.13>
  scripts/evaluate_puffer_fixed_door_checkpoint.py --checkpoint
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.bin
  --lineage-report
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.report.json
  --puffer-root
  <pufferlib-root>
  --agents 128 --steps 12000 --seed 20011 --live-yaw-cap-challenge
  --output
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.promotion-seed20011.json
  --skip-build`.
- Runtime/seeds: 1,589.80 s; evaluation seed 20011, temporal ablation seed
  20260734. The preceding 20-step native-build smoke rebuilt under the
  canonical Python 3.13 runtime after the project venv lacked
  `rich_argparse`; native extension SHA is `04735a28...94bb`.
- Artifacts: promotion report SHA
  `586f7daa0008b4a8ace8ca84cd63640a7b1e6903b3bbeb4f3c425c1469179e23`.
- Metrics: full camera 88.73% success (1,969/2,219), 77.67% outside-FOV
  success (859/1,106), and 0.27% collision (6/2,219). Masked camera was
  0.00% success and 0.00% collision over 1,152 episodes. Worst marginal was
  layout family 1 at 87.31% over 528 episodes. Policy-forward p95 was
  24.55 ms per 128-agent batch and closed-loop throughput was 4,989.60
  agent-steps/s. Resetting recurrence each step reduced success to 68.30% and
  increased collision to 6.13%; causal past-order scrambling was nearly
  neutral at 88.69% success. At the 8 deg/s yaw cap, success fell to 57.62%,
  outside-FOV success to 12.10%, and collision was 2.51%.
- Verdict: main simulation gate passes, recurrence is causally important, and
  the registered temporal-order intervention shows near-insensitivity rather
  than evidence that fine temporal order is used. The
  live-scale mismatch is now explicitly characterized: the capped policy fails
  the registered 70%/65% live-cap thresholds and therefore cannot advance to
  yaw authority.
- Next decision: finish the exact matched v59 seed-20011 evaluation before
  selecting a checkpoint. Regardless of winner, the next gate remains
  non-actuating shadow.

### 03:10-03:37 — P62 matched held-out comparator, v59

- Hypothesis: T61's gain survives a matched trained-bundle comparison against
  v59 on the same new evaluation stream and native binary.
- Parent checkpoint: v59 SHA `f676d12b...202cce`, trained from the same v53
  perception parent; its exact legacy action contract is 229.183 deg/s.
- One controlled variable: checkpoint plus its inseparable trained action
  contract changes from T61/70 deg/s to v59/229.183 deg/s. Evaluation seed,
  128 agents, 12,000 steps per condition, runtime stream/evidence-age
  contracts, policy architecture, and native extension SHA are identical.
- Command:
  `<python-3.13>
  scripts/evaluate_puffer_fixed_door_checkpoint.py --checkpoint
  artifacts/puffer_fixed_door_d1_v59_fresh_control_bc1m/flightrl_fixed_door_d1_seed11_1048576.bin
  --lineage-report
  artifacts/puffer_fixed_door_d1_v59_fresh_control_bc1m/flightrl_fixed_door_d1_seed11_1048576.report.json
  --puffer-root
  <pufferlib-root>
  --agents 128 --steps 12000 --seed 20011 --live-yaw-cap-challenge
  --output
  artifacts/puffer_fixed_door_d1_v59_fresh_control_bc1m/flightrl_fixed_door_d1_seed11_1048576.promotion-seed20011.json
  --skip-build`.
- Runtime/seeds: 1,597.98 s; evaluation seed 20011 and temporal seed 20260734.
- Artifacts: v59 held-out report SHA
  `441631c31468a4586f6f9e41cb8b140183ef361abe39fe88d424feedce7b5d0c`.
- Metrics: full camera 73.99% success (1,195/1,615; Wilson 95%
  71.80-76.07), 73.11% outside-FOV success (590/807), and 0.50% collision
  (8/1,615). Masked camera was 0.59% success but 12.49% collision over 1,185
  episodes. Worst marginal was layout family 1 at 64.36%. Policy p95 was
  24.82 ms and throughput 4,874.47 agent-steps/s. Recurrence reset reduced
  success by 44.36 points; temporal-order scrambling reduced it by 35.84.
  The 8 deg/s cap yielded 59.88% success, 17.86% outside-FOV, and 11.30%
  collision.
- Verdict: v59 fails this stricter held-out completion gate. Relative to the
  exact matched v59 run, T61 improves mission by 14.74 points, outside-FOV by
  4.56, worst marginal by 22.95, and masked collision by 12.49 while slightly
  reducing full collision. Candidate success Wilson 95% is 87.35-89.98%, with
  no overlap with v59.
- Next decision: run the preregistered selector before inspecting robustness
  challenges.

### 03:37-03:38 — P62 machine selection

- Hypothesis: T61 passes every preregistered held-out and three-seed screen
  without manual metric-direction mistakes.
- Parent checkpoints: candidate T61 seed 11 SHA `364647c6...8946`; baseline
  v59 SHA `f676d12b...202cce`; common v53 perception parent
  `0e831420...815761`.
- One controlled variable: none; read-only selection over frozen evidence.
- Command:
  `.venv/bin/python scripts/select_puffer_fixed_door_checkpoint.py
  --candidate-checkpoint
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.bin
  --candidate-report
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.promotion-seed20011.json
  --baseline-checkpoint
  artifacts/puffer_fixed_door_d1_v59_fresh_control_bc1m/flightrl_fixed_door_d1_seed11_1048576.bin
  --baseline-report
  artifacts/puffer_fixed_door_d1_v59_fresh_control_bc1m/flightrl_fixed_door_d1_seed11_1048576.promotion-seed20011.json
  --screen-seed11-report
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.report.json
  --screen-seed23-report
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed23/flightrl_fixed_door_d1_seed23_1048576.report.json
  --screen-seed47-report
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed47/flightrl_fixed_door_d1_seed47_1048576.report.json
  --output
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/promotion-selection-seed20011.json`.
- Runtime/seeds: 0.79 s; read-only, no new seed.
- Artifacts: selection report SHA
  `ab3ca1686740466cd175b8e48806763bfe7ea5f80d384fa7937ef4fb02587d92`.
- Metrics: every primary selection and seed 11/23/47 check passed; latency
  ratio was 0.989 and throughput ratio 1.024. `selection_passed=true`,
  recommended checkpoint is T61 seed 11, and `next_gate=shadow_only`.
  Separately, `live_cap_simulation_ready=false` because capped mission and
  outside-FOV miss 70%/65%.
- Verdict: T61 is the strongest simulation policy and the preregistered
  research recommendation. It is not a complete live-gate promotion.
- Next decision: preserve the user's exact v59 non-actuating first-morning
  command; use T61 only in an additional separately named non-actuating shadow
  after the v59 reference trace is valid.

### 03:38-03:43 — R63 fixed-dark lighting challenge

- Hypothesis: T61 retains useful semantic search under one fixed dark exposure.
- Parent checkpoint: T61 seed 11 SHA `364647c6...8946`.
- One controlled variable: native target camera mean is fixed to 20 instead of
  the baseline 18-110 range; no other challenge is active.
- Command:
  `<python-3.13>
  scripts/evaluate_puffer_fixed_door_checkpoint.py --checkpoint
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.bin
  --lineage-report
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.report.json
  --puffer-root
  <pufferlib-root>
  --agents 128 --steps 12000 --seed 20011 --challenge fixed-dark
  --control-report
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.promotion-seed20011.json
  --output
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.fixed-dark.challenge-seed20011.json
  --skip-build`.
- Runtime/seeds: 314.69 s; seed 20011, 128 agents, 12,000 steps.
- Artifacts: report SHA
  `9929fd15e31e63df0b3c5bc599364d5fac36827e4ec12e8425156bc23b9a0bd8`.
- Metrics: 94.68% success over 2,237 episodes (+5.95 points), 90.39%
  outside-FOV (+12.73), 1.03% collision (+0.76), worst marginal 91.11%.
- Verdict: no fixed-dark regression in this synthetic renderer; the gain is
  diagnostic and does not establish real low-light detector robustness.
- Next decision: keep real-shadow exposure/blur review as a separate gate.

### 03:43-03:48 — R64 obstacle-present challenge

- Hypothesis: the semantic policy is not obstacle-safe because obstacles were
  absent from training.
- Parent checkpoint: T61 seed 11.
- One controlled variable: probability of one route-intersecting cuboid changes
  from 0 to 1.
- Command:
  `<python-3.13>
  scripts/evaluate_puffer_fixed_door_checkpoint.py --checkpoint
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.bin
  --lineage-report
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.report.json
  --puffer-root
  <pufferlib-root>
  --agents 128 --steps 12000 --seed 20011 --challenge obstacle-present
  --control-report
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.promotion-seed20011.json
  --output
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.obstacle-present.challenge-seed20011.json
  --skip-build`.
- Runtime/seeds: 312.57 s; seed 20011, 128 agents, 12,000 steps.
- Artifacts: report SHA
  `048d4c623da759efa67f52d7c4f1bf7150d882dfb13c7fc1833d6ef2aad4025e`.
- Metrics: 3.33% success over 1,410 episodes (-85.40 points), 2.48%
  outside-FOV (-75.19), 30.07% collision (+29.80), worst marginal 2.25%.
- Verdict: catastrophic fail. Translation authority is prohibited; this policy
  is not an obstacle-aware navigator.
- Next decision: obstacle-aware training must be a separate future experiment,
  not patched into the morning demo.

### 03:48-03:53 — R65 1.2x room-footprint challenge

- Hypothesis: a modest room-size shift causes limited rather than catastrophic
  degradation.
- Parent checkpoint: T61 seed 11.
- One controlled variable: horizontal room bounds scale from +/-2.0 m to
  +/-2.4 m; topology and horizon remain fixed.
- Command:
  `<python-3.13>
  scripts/evaluate_puffer_fixed_door_checkpoint.py --checkpoint
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.bin
  --lineage-report
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.report.json
  --puffer-root
  <pufferlib-root>
  --agents 128 --steps 12000 --seed 20011
  --challenge room-footprint-1p2x --control-report
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.promotion-seed20011.json
  --output
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.room-footprint-1p2x.challenge-seed20011.json
  --skip-build`.
- Runtime/seeds: 311.80 s; seed 20011, 128 agents, 12,000 steps.
- Artifacts: report SHA
  `2c34f280e2bc8df1e9e66e6153ab00494a64b4c655af91882a4bd4054b17704a`.
- Metrics: 85.40% success over 1,932 episodes (-3.33 points), 73.54%
  outside-FOV (-4.13), 0.47% collision (+0.20), worst marginal 82.40%.
- Verdict: moderate generalization regression, not a collapse.
- Next decision: keep additional-room/topology claims out of the demo.

### 03:53-04:00 — R66 actor pixel-noise challenge

- Hypothesis: quantized camera noise exposes a limited raster sensitivity.
- Parent checkpoint: T61 seed 11.
- One controlled variable: actor camera channels receive deterministic
  sigma-8.5/255 noise followed by 17-level requantization; delta/motion are
  recomputed while detector evidence remains clean.
- Command:
  `<python-3.13>
  scripts/evaluate_puffer_fixed_door_checkpoint.py --checkpoint
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.bin
  --lineage-report
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.report.json
  --puffer-root
  <pufferlib-root>
  --agents 128 --steps 12000 --seed 20011 --challenge pixel-noise
  --control-report
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.promotion-seed20011.json
  --output
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.pixel-noise.challenge-seed20011.json
  --skip-build`.
- Runtime/seeds: 378.14 s; environment seed 20011, noise seed 20260735,
  128 agents, 12,000 steps.
- Artifacts: report SHA
  `e80bb03dabff995baa04be129ed4b5104ce0a2beda70ff79c9f42be4e2282c0a`.
- Metrics: 86.09% success over 2,013 episodes (-2.64 points), 73.59%
  outside-FOV (-4.07), 0.50% collision (+0.23), worst marginal 83.78%.
- Verdict: moderate actor-raster sensitivity; this is not end-to-end detector
  noise evidence.
- Next decision: require the real shadow detector/camera gate.

### 04:02-04:08 — R67 fixed camera-latency challenge

- Hypothesis: an additional 92.3 ms camera-derived-channel delay causes only a
  bounded regression.
- Parent checkpoint: T61 seed 11.
- One controlled variable: current camera/phase/evidence bundle is delayed six
  65 Hz control steps; current telemetry and previous action remain aligned.
- Command:
  `<python-3.13>
  scripts/evaluate_puffer_fixed_door_checkpoint.py --checkpoint
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.bin
  --lineage-report
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.report.json
  --puffer-root
  <pufferlib-root>
  --agents 128 --steps 12000 --seed 20011
  --challenge camera-latency-92ms --control-report
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.promotion-seed20011.json
  --output
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.camera-latency-92ms.challenge-seed20011.json
  --skip-build`.
- Runtime/seeds: 332.35 s; seed 20011, 128 agents, 12,000 steps.
- Artifacts: report SHA
  `8aba9865b793c274dcbcf9fd4c9dfe38b3df98f534ac8523fd832ded57d004e4`.
- Metrics: 87.54% success over 2,071 episodes (-1.19 points), 76.19%
  outside-FOV (-1.48), 0.39% collision (+0.12), worst marginal 85.79%.
- Verdict: no large regression in this descriptive diagnostic. No quantitative
  threshold was preregistered, and it does not model jitter, drops, or
  asynchronous transport.
- Next decision: stop adding versions. The remaining blockers are real shadow,
  8 deg/s authority feasibility, and obstacles; PPO would not isolate any of
  these and is not run.

### 04:08-04:25 — Step-back audit and final live-contract dry runs

- Hypothesis: the frozen evidence can support a morning handoff only if an
  independent reconstruction reaches the same selection and the newest live
  safety/output contracts fail closed without touching hardware.
- Parent checkpoints: unchanged T61 seed 11 SHA
  `364647c6982aebf064c12ff4f5c1a7bcec3c60f09640fe28529415d512298946`
  and v59 SHA
  `f676d12b9d37c27f4cc62f99beceec8f30e74c88be8564cb242c23755e202cce`.
- One controlled variable: none; this was read-only reconstruction plus
  synthetic monitor-mode contract validation. No new policy version or
  simulator treatment was introduced.
- Commands:
  `.venv/bin/pytest -q tests/test_puffer4_door*.py
  tests/test_evaluate_puffer_fixed_door_checkpoint_cli.py`;
  the T61 shadow command from the morning section with
  `--training-report` bound to its promotion report,
  `--output /tmp/door_puffer_shadow_t61_dryrun_v2_20260731.csv`, and
  `--dry-run`; then
  `.venv/bin/python scripts/build_fixed_door_live_readiness.py --checkpoint
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.bin
  --simulation-report
  artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.promotion-seed20011.json
  --shadow-summary
  /tmp/door_puffer_shadow_t61_dryrun_v2_20260731.summary.json --shadow-csv
  /tmp/door_puffer_shadow_t61_dryrun_v2_20260731.csv --output
  /tmp/fixed_door_t61_dryrun_readiness_v2_20260731.json`.
- Runtime/seeds: 10.22 s for 235 focused tests; each synthetic dry run took
  under one second and consumed no environment seed.
- Artifacts: live safety contract `fixed-door-yaw-only-live-v2`, SHA
  `0d80df47d24fc3a52025cf2c7a0c67a2c571536b68e870cdf4ca9d233d419e13`;
  `/tmp/door_puffer_shadow_t61_dryrun_v2_20260731.{csv,summary.json}` and
  `/tmp/fixed_door_t61_dryrun_readiness_v2_20260731.json`. These are synthetic
  diagnostics, not real shadow evidence.
- Metrics: 235/235 focused tests passed. The dry run bound the exact T61
  checkpoint, promotion, lineage, action, recurrent-policy, evidence-age,
  detector, hardware, and safety identities; translation and executed
  previous action were zero and projected yaw was capped at 8 deg/s. The
  rebuilt readiness reports `sim_live_yaw_cap_gate_passed=false`,
  `replay_yaw_gate_passed=false`, and `next_live_gate_passed=false`. A
  subsequent truth-model check found that this failed diagnostic still carried
  the contradictory label `approved_authority=yaw_only`; that label defect is
  repaired in the next entry.
- Verdict: independent reconstruction exactly reproduced the selector and all
  five challenge deltas. Re-running the selector produced a byte-identical
  report with SHA
  `ab3ca1686740466cd175b8e48806763bfe7ea5f80d384fa7937ef4fb02587d92`.
  A repeated v59 dry run to an existing output failed
  before inference; SHA-256 of both CSV and summary was identical before and
  after, proving shadow evidence is not overwritten. The audit also found that
  the detector contract pins model IDs but not immutable repository revisions
  or weight hashes, so it cannot authorize live control.
- Next decision: freeze policy experiments. Keep the heartbeat active, resolve
  only truth-model/documentation defects, re-run focused verification after
  any such repair, and do not manufacture additional versions without a new
  causal question.

### 04:25-04:32 — A0 readiness authority truth-model repair

- Hypothesis: if effective authority fields are derived from the gate boolean
  at one typed boundary, a failed diagnostic can retain the candidate being
  evaluated without falsely claiming that any axis is approved.
- Parent checkpoints: unchanged T61 and v59 bytes; no report or checkpoint was
  rewritten.
- One controlled variable: readiness schema changes from independent
  `approved_authority`/axis literals to schema v2 derived decision fields.
  Candidate authority remains diagnostic metadata; effective authority is
  `none` with all axes false unless `next_live_gate_passed=true`.
- Commands: red
  `.venv/bin/pytest -q tests/test_semantic_readiness_authority.py`; green
  `.venv/bin/pytest -q tests/test_semantic_readiness_authority.py
  tests/test_puffer4_door_readiness.py tests/test_semantic_yaw_authority.py
  tests/test_semantic_bounded_authority.py`; full
  `.venv/bin/pytest -q tests/test_puffer4_door*.py
  tests/test_evaluate_puffer_fixed_door_checkpoint_cli.py`; then the exact
  readiness-builder command above to new output
  `/tmp/fixed_door_t61_dryrun_readiness_schema2_20260731.json`.
- Runtime/seeds: expected red 5 failed/1 passed; green 6/6; authority/readiness
  regression 42/42 in 1.70 s; full fixed-door 235/235 in 9.85 s. Deterministic
  tests and synthetic dry-run evidence; no environment seed.
- Artifacts: new 93-LOC
  `src/flightrl/semantic/readiness_authority.py`, schema-v2 builders/loaders,
  and 151-LOC focused tests. The synthetic schema-v2 diagnostic preserves
  `candidate_authority=yaw_only` but records `approved_authority=none`, every
  effective axis false, and `next_live_gate_passed=false`.
- Metrics: Ruff and `py_compile` pass; all touched source/test files remain
  below 300 LOC. Loading the failed T61 diagnostic through the exact control
  loader raises `next-live readiness gate did not pass`.
- Verdict: safety truth-model pass. There was no actuation bypass before the
  repair because the loader already rejected the false gate, but reports can no
  longer contradict that decision. Passing immutable schema-v1 artifacts
  remain compatible; failed or forged v2 authority combinations fail closed.
- Next decision: freeze source again and independently reconstruct the complete
  observation/action/evaluation chain before the final morning audit.

### 04:32-04:39 — A0 complete contract reconstruction

- Hypothesis: an end-to-end reconstruction will either prove the frozen v59/T61
  observation, recurrence, previous-action, yaw, checkpoint, and evaluation
  chain or identify a concrete unsupported path before the morning handoff.
- Parent checkpoints: exact unchanged v59 and T61 seed-11 bytes.
- One controlled variable: none; this was an independent read-only contract
  audit, not a policy treatment.
- Command:
  `.venv/bin/pytest -q tests/test_puffer4_door_observation.py
  tests/test_puffer4_door_policy_contract.py
  tests/test_puffer4_door_runtime.py tests/test_puffer4_door_action.py
  tests/test_puffer4_door_contract.py tests/test_puffer4_door_bundle.py
  tests/test_puffer4_door_shadow_io.py
  tests/test_puffer4_door_shadow_projection.py
  tests/test_puffer4_door_promotion_eval.py`.
- Runtime/seeds: 1.13 s; 51 deterministic tests, no environment seed.
- Artifacts: no new checkpoint/report. The audit re-read the immutable v59
  action SHA `e666cf9c...879a80b5`, corrected action SHA
  `55fc561c...93e27c`, and historical policy SHA
  `ad6fa58f...8ef8058`.
- Metrics: 51/51 passed. Numeric ABI, privileged isolation, host/native frame
  preprocessing, selected BC/DAgger terminal resets, executed previous action,
  yaw sign/scale, and strict off-center nonzero-yaw alignment all agree. The
  audit found two unsupported code paths: the generic PuffeRL PPO loop did not
  mask recurrent state after terminals, and policy-contract v1 described a
  lower-center self mask although both implementations use upper-corner
  wedges. Neither defect contaminated v59 or T61: both selected reset-aware
  bootstrap BC with zero PPO rollouts.
- Verdict: checkpoint evidence remains valid, but future PPO must fail closed
  and new policy metadata must be versioned truthfully.
- Next decision: repair each unsupported path independently without changing
  any existing report bytes or policy behavior.

### 04:39-04:45 — A0 policy-contract metadata v2

- Hypothesis: versioning only the false self-mask descriptor can make new
  metadata truthful while preserving exact historical v1 validation.
- Parent checkpoints: v59 and T61, both grandfathered to their exact v1 policy
  contract; checkpoint and report bytes remain unchanged.
- One controlled variable: new policy-contract reports change from v1 to v2
  and name the implemented upper-corner wedge mask. Observation indices,
  preprocessing, actor inputs, recurrence, actions, and runtime behavior do not
  change.
- Commands: focused
  `.venv/bin/pytest -q tests/test_puffer4_door_policy_contract.py`; load both
  real v59/T61 bundles; then the combined fixed-door command below.
- Runtime/seeds: focused 7/7 passed; deterministic contract tests, no
  environment seed.
- Artifacts: new default `fixed-door-recurrent-policy-v2`, SHA
  `284dda85a0b949b07e6a8aaa2a3a370ee75ab206ff2eba4a5ad5d40a52f6ce33`;
  exact v1 SHA
  `ad6fa58f50a1c0754d572643a9d7affe65f3e73d4d814c51030c733588ef8058`
  remains an approved immutable payload.
- Metrics: self-consistent unknown versions and mutated descriptors are
  rejected. Both real v59 and T61 reports still load with v1, and their report
  hashes remain unchanged.
- Verdict: metadata-only repair passes; it does not relabel historical
  artifacts or claim a new learned policy.
- Next decision: new training runs must emit v2; existing v1 evidence remains
  readable only as its exact approved payload.

### 04:41-04:49 — A0 yaw single-source hardening

- Hypothesis: making all yaw conversions derived from immutable unit-bearing
  contracts will reject the original 70 deg/s versus 4 rad/s class of defect
  before training and prove the final native rate.
- Parent checkpoints: unchanged v59 and T61 bytes; this is contract
  enforcement only.
- One controlled variable: action/live contract validation and derived
  conversion ownership. No numeric limit, policy tensor, simulator scene, or
  evaluation treatment changes.
- Commands: expected-red then green
  `.venv/bin/pytest -q tests/test_puffer4_door_contract.py
  tests/test_puffer4_door_action.py`; focused regression
  `.venv/bin/pytest -q tests/test_puffer4_door_contract.py
  tests/test_puffer4_door_action.py
  tests/test_puffer4_door_canonical_evaluation.py
  tests/test_puffer4_door_live_evidence.py`; export binding test; strict
  `cc -std=c11 -Wall -Wextra -Werror -c
  src/flightrl/native/native_door_action.c`; and the combined command below.
- Runtime/seeds: expected red 4 failures/13 passes; focused green 30/30 in
  0.96 s; deterministic tests, no environment seed.
- Artifacts: no checkpoint/report rewrite. Contract payloads and hashes are
  deliberately byte-stable: corrected action
  `55fc561c...93e27c`, legacy v59 `e666cf9c...879a80b5`, and live safety
  `0d80df47...d419e13`.
- Metrics: the action constructor rejects a declared policy rate above the
  physics ceiling and rejects a “legacy direct” mapping that is not exactly
  that ceiling. `native_yaw_action_scale` is the sole Python derivation of
  `radians(policy_deg_s) / physics_rad_s`; live-cap normalization is solely
  `DoorLiveSafetyContract.normalized_yaw_limit()`. The C golden test consumes
  the canonical contract, then settles the native physics rate for action
  `-0.5` at `-0.6108651 rad/s` (`-35 deg/s`), not `-2 rad/s`. The exported
  binding is asserted to call the mapper and now uses the named
  `SIXDOF_PHYS_MAX_RATE_YAW` index rather than magic index 7.
- Verdict: pass. Policy semantics, physics capacity, and live safety remain
  separate quantities with one owner each; impossible combinations and
  cross-language drift fail in focused tests.
- Next decision: do not broaden into a generated cross-language ABI tonight.
  Preserve the independent C conversion as a golden-tested boundary and keep
  checkpoint provenance immutable.

### 04:43-04:49 — A0 generic recurrent-PPO fail-closed guard

- Hypothesis: rejecting generic PPO before environment initialization prevents
  silent recurrent-state leakage without disturbing the reset-aware BC/DAgger
  path used by v59/T61.
- Parent checkpoints: unchanged v59 and T61; both used
  `total_timesteps=0` after bootstrap imitation.
- One controlled variable: positive generic fixed-door PPO budgets are
  disallowed locally until a terminal-masked rollout/training implementation
  has its own isolation test.
- Command:
  `uv run pytest -q tests/test_puffer4_door_ppo_guard.py
  tests/test_puffer4_door_training.py tests/test_puffer4_door_asymmetric.py`;
  then the combined fixed-door command below.
- Runtime/seeds: 20/20 focused tests in 0.64 s; no environment seed.
- Artifacts: 43-LOC `puffer4_door_training_gates.py` guard and 33-LOC focused
  regression test; no external PufferLib file was edited.
- Metrics: expected red reached the unsafe generic Puffer initializer; green
  raises the explicit terminal-mask error before argument mutation or
  Puffer/environment initialization. Zero-budget reset-aware BC/DAgger remains
  enabled. The separate asymmetric on-policy implementation already masks
  terminal state and is not affected.
- Verdict: pass; unsupported generic recurrent PPO now fails closed.
- Next decision: a future PPO experiment requires a separately reviewed
  terminal-masked implementation and two-episode hidden-state isolation test.

### 04:49 — Combined source freeze verification

- Command:
  `.venv/bin/pytest -q tests/test_puffer4_door*.py
  tests/test_evaluate_puffer_fixed_door_checkpoint_cli.py`.
- Runtime/seeds: 10.31 s; no environment seed.
- Metrics: 242/242 passed. Scoped Ruff, `py_compile`, and strict native-action C
  compilation pass. All touched source files are at or below 300 LOC.
- Verdict: source freeze passes. No drone/radio/hardware operation, flashing,
  or authority occurred.
- Next decision: update the morning briefing with these fail-closed semantics,
  then perform a final read-only audit rather than run another policy version.

### 04:50 — Final historical-bundle monitor-only revalidation

- Hypothesis: policy-contract v2 support must not invalidate or silently relabel
  the exact v1-bound v59/T61 artifacts used by the morning commands.
- Parent checkpoints: exact v59 and T61 seed-11 bytes.
- One controlled variable: current loader/contract code after source freeze;
  both policies and reports are unchanged.
- Commands: the exact v59 and T61 shadow commands in the morning section, each
  with `--dry-run` and a new
  `/tmp/door_puffer_shadow_{v59,t61}_final_contract_dryrun_20260731.csv`
  output.
- Runtime/seeds: 0.98 s combined; synthetic one-row inputs, no environment
  seed, cflib, radio, camera, or hardware initialization.
- Artifacts: two `/tmp` CSV/summary pairs; diagnostic only.
- Metrics: both bind exact v1 policy SHA `ad6fa58f...8ef8058`, exact checkpoint,
  report, action, evidence-age, detector, and hardware identities. Both report
  `monitor_only=true`, `controls_drone=false`, finite outputs, zero translation,
  zero executed previous action, and an 8 deg/s projected yaw ceiling.
- Verdict: pass. The one-row dry runs intentionally fail real-shadow coverage,
  timestamp, phase, and detection requirements and cannot be used as gate
  evidence.
- Next decision: retain the exact real-shadow-first operator sequence; no live
  control command is added.

### 04:50-04:56 — Independent morning-handoff audit

- Hypothesis: a fresh reader can reconstruct the recommendation and find no
  path from the documented commands to unreviewed authority.
- Parent checkpoints: exact v59 and T61 seed-11 bundles.
- One controlled variable: none; final read-only evidence/command audit.
- Commands: re-read artifact JSON/hashes, run the exact parser/import path,
  inspect shadow camera/telemetry capture for commander calls, check morning
  output nonexistence, and rerun the 242-test fixed-door suite.
- Runtime/seeds: 9.88 s for tests; no environment seed.
- Artifacts: no new policy/evaluation artifact.
- Metrics: 242/242 tests pass. All briefing hashes, metrics, flags, and lineages
  match; v59 decodes only as `grandfathered_v59`, T61 as `promotion_v3`; both
  operator outputs are unused. The real shadow path opens camera and read-only
  telemetry, passes zero executed-action history, and contains no
  commander/control call. No live-control command appears.
- Verdict: no blocker. The audit found only a prose-completeness gap, now
  repaired: the stop list explicitly includes at least 50 rows,
  grounding-result frame order, zero projected translation/previous action,
  and the bound 8 deg/s projection.
- Next decision: hold the evidence freeze and continue heartbeat audits; do not
  add another policy treatment without a new causal question.

### 04:56-05:02 — A0 preliminary single-file detector binding (superseded)

- Hypothesis: a model-ID-only detector contract can silently load different
  weights; binding exact repository commits and weight bytes can close that
  provenance gap without changing the policy or authority decision.
- Parent checkpoints: exact unchanged v59 and T61 seed-11 bundles.
- One controlled variable: fixed-door shadow/control detector artifact
  provenance changes from model IDs only to full commit plus one weight
  SHA-256.
  Prompt, thresholds, preprocessing, device, policy, and gate thresholds do not
  change.
- Commands: inspect local Hugging Face `refs/main` and snapshot symlinks;
  `sha256sum` both cached weight files; expected-red then green
  `.venv/bin/pytest -q
  tests/test_puffer4_door_shadow_detector_contract.py
  tests/test_semantic_model_artifact.py`; focused six-file detector/identity/
  readiness regression; construct both pinned models once on CPU; then
  `.venv/bin/pytest -q tests/test_puffer4_door*.py
  tests/test_evaluate_puffer_fixed_door_checkpoint_cli.py
  tests/test_semantic_model_artifact.py tests/test_semantic_grounding.py`.
- Runtime/seeds: expected red failed collection because the verified-artifact
  module did not yet exist; focused 53/53 in 3.97 s; Grounding DINO/CLIP CPU
  construction 3.21/2.10 s; preliminary 256/256 in 9.97 s. The exact external
  `uv run` interpreter reports MPS built/available; combined pinned-model MPS
  construction took 4.36 s. No environment seed.
- Artifacts: `fixed-door-real-shadow-detector-v2`, SHA
  `9adb10d73b047bcaa7fce2a02b205a975077a6d784decb4cbd40900de830424a`.
  Grounding DINO commit
  `a2bb814dd30d776dcf7e30523b00659f4f141c71`, file
  `model.safetensors`, SHA
  `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3`;
  CLIP commit `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, file
  `pytorch_model.bin`, SHA
  `a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f`.
- Metrics: the preflight resolves exact commits with `local_files_only=true`
  and verifies the advertised weight digest. Both installed Transformer
  models load successfully from those revisions; all touched files remain
  below 300 LOC; scoped Ruff and `py_compile` pass. After wiring the same pins
  into the dormant control path, the 256-test rerun passed in 9.97 s. A
  synthetic 128x96 MPS check measured one cold Grounding DINO pass at
  903-1,105 ms, then five warmed passes at p95 314.80 ms; the separately forced
  CLIP-verifier path warmed to p95 19.70 ms. This is runtime feasibility only,
  not real-camera latency evidence.
- Verdict: superseded after independent review. The verified cache path was
  discarded and Transformers resolved the repository again; ancillary
  configuration/tokenizer bytes, alternate-weight precedence, and the
  verify-then-reopen interval were not bound. The pristine cache happened to
  select the advertised files, so no mismatch was observed, but v2 was not
  promotion-grade immutable provenance. A first attempt to pipe the shadow
  script's JSON plus trailing path lines directly into `jq` also returned
  parse exit 5 after the script had safely completed.
- Next decision: hold the source freeze and replace v2 with a complete private
  manifest snapshot before admitting any real shadow evidence.

### 05:09-09:12 — A0 detector-v3 private-manifest repair and closeout

- Hypothesis: copying every approved model/processor/tokenizer artifact into a
  clean private snapshot, verifying bytes during that copy, and loading only
  that path will bind actual inference semantics rather than merely checking
  an adjacent cache file.
- Parent checkpoints: exact unchanged v59 and T61 seed-11 bundles.
- One controlled variable: detector artifact loading changes from v2's
  verify-then-re-resolve path to a complete v3 manifest/private snapshot.
  Prompt, thresholds, preprocessing, device, policy, and all authority
  decisions remain unchanged.
- Commands: independent read-only loader-precedence and cache audit; expected
  red `.venv/bin/pytest -q tests/test_semantic_model_snapshot.py`; SHA-256
  inventory of both exact Hub snapshots; focused snapshot/loader/contract
  tests; scoped Ruff and `py_compile`; software-only CPU and external-MPS model
  construction; exact v59 CLI `--dry-run`; and
  `.venv/bin/pytest -q tests/test_puffer4_door*.py
  tests/test_evaluate_puffer_fixed_door_checkpoint_cli.py
  tests/test_semantic_model_artifact.py
  tests/test_semantic_model_snapshot.py tests/test_semantic_grounding.py`.
- Runtime/seeds: initial red stopped during collection because
  `weights_format` did not yet exist; focused final 23/23 in 0.65 s; full
  fixed-door/semantic gate 264/264 in 10.76 s. Pinned DINO and CLIP both
  constructed successfully through the centralized factory on Apple MPS. No
  experiment seed and no physical hardware access.
- Artifacts: `fixed-door-real-shadow-detector-v3`, SHA
  `75d80525b6b532c365c2df9b7642ef73d81b9b51017efc2e6c06cc9ce603a1ad`;
  eight-file Grounding DINO manifest at commit
  `a2bb814dd30d776dcf7e30523b00659f4f141c71`; eight-file CLIP manifest
  at commit `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`; dry-run outputs
  `/tmp/door_puffer_shadow_v59_detector_v3_dryrun_20260731_0518.csv`
  and matching `.summary.json`.
- Metrics: the private directory contains only manifest files, excludes an
  injected alternate weight, remains unchanged after source-cache mutation,
  and is removed after load. DINO derives `use_safetensors=true`; CLIP derives
  `false` from the typed contract rather than a duplicate constant. Both pass
  `local_files_only=true`, `trust_remote_code=false`, and
  `weights_only=true`. Runtime identity binds Transformers 5.14.1, Torch
  2.13.0, tokenizers 0.22.2, safetensors 0.8.0, NumPy 2.5.1, Pillow 12.3.0,
  and huggingface-hub 1.24.0. The v59 dry run is monitor-only, zero
  translation, and binds detector-v3 identity SHA
  `2b23332781069649c3f5355872fdbe40df0339d9b30f8264f76adb7e1a8b5fb4`.
- Verdict: pass for source-level detector provenance and morning shadow
  construction; this grants no flight authority. The correct gate remains
  shadow-only because T61 still fails the separate 8 deg/s simulation gate.
- Next decision: stop experimental work. The heartbeat was paused at 09:12
  CEST when the delayed closeout turn observed that the 07:30 deadline had
  passed; no hardware or radio path was entered during the overrun.

## Registered experiments

### A0 — Static Gate A and yaw-contract mechanism audit

- Hypothesis: the declared 70 deg/s door yaw limit is ignored by the native
  action path, and the evidence chain needs stricter hash/freshness binding.
- Primary variable: none; read-only audit followed by focused contract tests.
- Frozen invariants: v59 bytes, firmware-stabilized setpoint boundary,
  translation disabled, 8 deg/s future live cap, no hardware access.
- Primary metric: exact unit/sign/scale trace from normalized policy action to
  executed yaw and previous-action observation.
- Kill condition: any ambiguous sign, unbound evidence, non-finite path, or
  stale-input path keeps live authority disabled.
- Expected runtime: under 45 minutes.

### T60-YAW — Corrected native yaw-scale BC

- Hypothesis: executing normalized yaw at the declared 70 deg/s scale, with the
  previous-action observation equal to the executed normalized action, will
  preserve or improve fresh-controller BC mission behavior relative to v59.
- Baseline: exact v59 checkpoint and corrected reevaluation above.
- Primary intervention: native door yaw execution scale changes from inherited
  229.18 deg/s to declared 70 deg/s. No DAgger, PPO, obstacles, room-count
  change, or camera randomization.
- Seeds: train `11` first; then `23` and `47` if the mechanism and seed-11
  screen pass. Evaluation streams are frozen and disjoint from train streams.
- Budget: 1,048,576 transitions per seed; 128 agents; horizon 64; 128 updates;
  learning rate 0.001; policy roll-in 0; PPO rollouts 0.
- Parent: v53 perception tensors only, SHA above; fresh fusion, recurrence, and
  decoder.
- Scene/held-out bindings: native environment name
  `flightrl_fixed_door_d1`; exact scene implementation and generated config are
  content-hashed in each run report. T60 used train/evaluation seeds 11/10011.
  The initial registration lacked a separate human-readable
  scene-distribution ID, which is a recorded protocol gap; later T61 reports
  additionally bind the episode-stream contract SHA
  `9d2db7809e76b8fa103baa94587f9c7e36f4813655fa97a06c575fe31ec96c4b`.
- Observation/action contracts: recurrent observation/MinGRU contract SHA
  `ad6fa58f50a1c0754d572643a9d7affe65f3e73d4d814c51030c733588ef8058`;
  corrected action contract SHA
  `55fc561c3fafacda47c950cb65398e657d1cfad26d33ce3635c57e72ad93e27c`.
- Primary metric: full-camera mission success. Risk metrics: collision,
  outside-FOV success, masked-camera success, authority-capped success, and a
  held-out appearance/layout perturbation.
- Pilot/screen threshold: not worse than v59 by more than 2 completion points,
  collision at most 3%, outside-FOV at least 65%, masked success at most 10%.
  Actual promotion still requires at least +5 completion points plus all
  registered camera-causality, collision, worst-group, and runtime checks.
- Kill condition: teacher gate regression, sign/scale mechanism failure,
  collision above 3%, or camera-causality failure.
- Expected runtime: 10-25 minutes per seed plus evaluation.

### T60-DAGGER — Fresh-controller on-policy aggregation

- Hypothesis: scheduled student roll-in exposes and corrects fresh-controller
  distribution-shift failures, improving completion by at least five points
  over corrected-yaw BC at the same sample budget.
- Baseline: T60-YAW BC with identical seed, parent, scene stream, optimizer,
  agents, horizon, updates, and evaluation streams.
- Primary intervention: maximum policy roll-in fraction changes from 0.0 to
  0.5; teacher labels remain privileged only during training.
- Seeds: `11` first; seeds `23` and `47` only if seed 11 clears the
  preregistered gain criterion.
- Budget: exactly 1,048,576 environment transitions per seed.
- Frozen scene/held-out bindings: exactly the matched BC environment name,
  source/build manifest, `fixed-door-episode-stream-v1` SHA
  `9d2db7809e76b8fa103baa94587f9c7e36f4813655fa97a06c575fe31ec96c4b`,
  and the same per-seed held-out evaluation stream.
- Frozen observation/action contracts: SHA
  `ad6fa58f50a1c0754d572643a9d7affe65f3e73d4d814c51030c733588ef8058`
  and
  `55fc561c3fafacda47c950cb65398e657d1cfad26d33ce3635c57e72ad93e27c`,
  respectively.
- Promotion threshold: at least +5 completion points over matched BC,
  collision at most 3%, no outside-FOV/worst-group or camera-causality
  regression.
- Kill condition: collision above 3%, masked success above 10%, or no matched
  seed-11 gain.
- Expected runtime: 10-25 minutes per seed plus evaluation.

## Morning briefing

### Outcome and recommendation

- Strongest tested simulation bundle: T61 seed 11,
  `artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.bin`,
  SHA-256
  `364647c6982aebf064c12ff4f5c1a7bcec3c60f09640fe28529415d512298946`.
- Exact lineage: v53 perception-only parent SHA
  `0e831420c7d1d1a1a46979aa1e096122cf511bb8dfd51cd12510fbb94d815761`
  -> fresh seed-11 fusion/MinGRU/decoder -> corrected 70 deg/s action contract
  -> episode-indexed pure BC, 1,048,576 transitions. Lineage report SHA
  `0151023e90a75b9ee54f8e132d4c6af229a324e43468bdb7a4709179dea8457e`;
  held-out promotion report SHA
  `586f7daa0008b4a8ace8ca84cd63640a7b1e6903b3bbeb4f3c425c1469179e23`;
  selector SHA
  `ab3ca1686740466cd175b8e48806763bfe7ea5f80d384fa7937ef4fb02587d92`.
- Gate verdict: **shadow-only**. T61 is a research/simulation recommendation,
  not a complete live promotion. No yaw-only or bounded-forward authority is
  authorized, even after a good shadow, because the 8 deg/s simulation gate
  already fails. Firmware stabilization, takeoff, height/position hold,
  landing, and abort remain the only flight authority.
- Operational reference/rollback checkpoint: v59 at the frozen path above,
  SHA
  `f676d12b9d37c27f4cc62f99beceec8f30e74c88be8564cb242c23755e202cce`.
  “Rollback” means zero learned authority and firmware hold/land/abort; it
  does not mean flying v59.
- Overnight safety record: no physical drone or radio connection, command,
  arming, motion, flight, flashing, or live authority occurred. All shadow
  executions were explicitly synthetic `--dry-run` checks.

### Comparable evidence

| Evidence | Mission | Outside FOV | Collision | Masked success / collision | Worst layout |
|---|---:|---:|---:|---:|---:|
| v59 authoritative reevaluation, seed 10011 | 79.11% | 74.73% | 0.56% | 1.53% / not reported | not reported |
| v59 matched promotion, seed 20011 | 73.99% | 73.11% | 0.50% | 0.59% / 12.49% | 64.36% |
| T61 matched promotion, seed 20011 | 88.73% | 77.67% | 0.27% | 0.00% / 0.00% | 87.31% |
| T61 at 8 deg/s | 57.62% | 12.10% | 2.51% | n/a | n/a |
| v59 at 8 deg/s | 59.88% | 17.86% | 11.30% | n/a | n/a |

The 8 deg/s simulation challenge changes yaw clipping only and deliberately
leaves the learned forward action unchanged, so it isolates the yaw-scale
effect. It is a necessary precondition, not a simulation of the morning
yaw-only command; any eventual physical yaw-only gate still hard-zeros all
translation.

The matched fixed-step stream uses the same seed, build, procedural contract,
agents, and steps. It is not the same realized episode set: policy-dependent
episode lengths produced 2,219 T61 and 1,615 v59 full-camera completions.
Therefore the +14.74-point T61 delta is a trained-bundle comparison; weights
and the inseparable trained action contract change together, so it is not
attributable to weights alone or to yaw repair alone.

T61 training screens were 89.64%/79.78%/0.30% for seed 11,
87.03%/74.54%/0.37% for seed 23, and 89.93%/80.08%/0.49% for seed 47
(mission/outside-FOV/collision). Their 88.87% mission mean versus a single
v59 result is descriptive, not a matched three-seed baseline estimate.
DAgger seed 11 was worse than matched pure BC by 1.65 mission points and
increased collision; seeds 23/47 were not run because seed 11 hit the
preregistered kill condition.

Recurrence reset reduced T61 mission from 88.73% to 68.30%, while the
registered temporal-order scramble was nearly neutral at 88.69%. This means
the recurrent state is important, but this specific ablation did not show
dependence on fine temporal order. Policy-forward p95 was 24.55 ms per
128-agent batch and closed-loop throughput was 4,989.60 agent-steps/s.

Separate diagnostic challenges changed only one registered factor:

- fixed synthetic darkness: 94.68% mission, 1.03% collision; no real low-light
  claim;
- one route obstacle: 3.33% mission, 30.07% collision; catastrophic;
- 1.2x room footprint: 85.40% mission, 0.47% collision; this is not a new room
  topology;
- actor pixel noise: 86.09% mission, 0.50% collision; detector stayed clean;
- fixed 92 ms camera-channel delay: 87.54% mission, 0.39% collision; no jitter
  or transport claim.

PPO was not run: DAgger already rejected the distribution-shift hypothesis,
while the decisive failures are action-cap feasibility, obstacles, and missing
real evidence. Changing optimizer would not isolate any of them.

### Verification executed

- Final central fixed-door command:
  `.venv/bin/pytest -q tests/test_puffer4_door*.py
  tests/test_evaluate_puffer_fixed_door_checkpoint_cli.py
  tests/test_semantic_model_artifact.py
  tests/test_semantic_model_snapshot.py tests/test_semantic_grounding.py` ->
  264 passed in 10.76 s after the final source change.
- Authority/readiness schema-v2 regression -> 42 passed; its new focused file
  was first observed red at 5 failed/1 passed, then green at 6/6.
- Ruff, `py_compile`, strict C11 `-Wall -Wextra -Werror`, native CPU build,
  vector reset/RNG smokes, and <=300-LOC checks passed in their recorded
  chronology stages.
- Exact seed-11/23/47 BC, seed-11 DAgger, held-out T61/v59, selector, and five
  challenge commands are recorded above with runtime, seed, artifact, and
  SHA-256. Rebuilt selector bytes exactly match the stored selection report.
- Synthetic v59 and T61 shadow dry runs bound monitor-only identity and the
  8 deg/s projection. Reusing a v59 output path failed before modification and
  left both evidence hashes unchanged. The failed T61 schema-v2 readiness
  artifact grants no effective authority and is rejected by the control
  loader.

### Single sources of truth

The repair deliberately does not create one global “yaw” number. It creates
one immutable owner for each different physical meaning:

- `fixed-door-declared-yaw-v1`, SHA
  `55fc561c3fafacda47c950cb65398e657d1cfad26d33ce3635c57e72ad93e27c`,
  owns normalized policy actions, 70 deg/s semantics, sign, action order, the
  4 rad/s simulator ceiling, and executed previous-action feedback;
- `fixed-door-recurrent-policy-v1`, SHA
  `ad6fa58f50a1c0754d572643a9d7affe65f3e73d4d814c51030c733588ef8058`,
  remains the exact historical v59/T61 owner of all 9,248 observation fields
  and MinGRU/reset semantics. New reports use truthful
  `fixed-door-recurrent-policy-v2`, SHA
  `284dda85a0b949b07e6a8aaa2a3a370ee75ab206ff2eba4a5ad5d40a52f6ce33`,
  which changes only the implemented self-mask description;
- `fixed-door-yaw-only-live-v2`, SHA
  `0d80df47d24fc3a52025cf2c7a0c67a2c571536b68e870cdf4ca9d233d419e13`,
  owns the separate 8 deg/s cap, 0.75 s staleness, 0.20-0.80 m height,
  15 s duration, and translation-disabled envelope;
- versioned episode-stream and evidence-age contracts own their independent
  reset and time semantics.
- schema-v2 readiness uses one typed decision helper: the gate boolean derives
  effective authority and every axis. A failed gate may name
  `candidate_authority=yaw_only` for diagnostics, but it must serialize
  `approved_authority=none` and all axes false.

Native config is generated from these types and verified again at load time.
Training, checkpoint lineage, evaluation, shadow, readiness, and control bind
the exact contract IDs and SHA-256 digests. Unknown, locally rehashed, stale,
or cross-lineage contracts fail closed. Detector v3, SHA
`75d80525b6b532c365c2df9b7642ef73d81b9b51017efc2e6c06cc9ce603a1ad`,
owns exact Grounding DINO and CLIP commits, complete eight-file manifests,
weight formats, and loader runtime versions. A single factory used by shadow
and dormant control copies those bytes into clean private snapshots and makes
the typed format the sole source of the Transformers weight-selection flag.

The action owner also derives the only host-side
`radians(policy_deg_s) / physics_rad_s` scale and rejects any policy rate above
the physics ceiling. The live owner derives the only normalized 8-deg/s cap.
The native golden test sources both inputs from that canonical action object,
verifies the executed previous-action round trip, and rate-settles to the
declared physical value. The exported binding uses the named physics-yaw index.
Generic fixed-door PuffeRL PPO is now explicitly unsupported and fails before
initialization because its recurrent state is not terminal-masked; this did not
affect the zero-rollout BC/DAgger checkpoints evaluated tonight.

### World Action Model feasibility

The separate
[primary-source audit](flightrl_world_action_models_20260730.md) finds WAMs
research-relevant but infeasible for this live path. WorldFly is the closest
UAV analogue, yet it is simulation-only, uses metre-scale/30-degree action
primitives, and degrades sharply on unseen intersections. Current navigation
and robot WAMs are generally billion-parameter, datacenter-GPU systems far
below the 65 Hz edge loop. EgoWAM/NavWAM do support one plausible later test:
a training-only action-conditioned next-latent objective, removed at inference.
That experiment should run only after a measured representation failure and
must not change the morning gate, firmware boundary, or actor interface.

### Exact morning operator sequence

The first command remains exactly the requested non-actuating v59 reference:

```bash
uv run python scripts/crazyflie_door_puffer_shadow.py \
  --checkpoint artifacts/puffer_fixed_door_d1_v59_fresh_control_bc1m/flightrl_fixed_door_d1_seed11_1048576.bin \
  --training-report artifacts/puffer_fixed_door_d1_v59_fresh_control_bc1m/flightrl_fixed_door_d1_seed11_1048576.reevaluation.json \
  --prompt "interior door" \
  --threshold 0.25 \
  --duration-s 20 \
  --output artifacts/crazyflie_logs/door_puffer_shadow_v59_gateA_run1.csv
```

This command cannot actuate the drone. Its legacy reevaluation binding makes
it a reference trace only: the fixed-door readiness builder requires a
promotion-v3 report. If sampled coverage is below 20 s, preserve run 1 and use
a new path:

```bash
uv run python scripts/crazyflie_door_puffer_shadow.py \
  --checkpoint artifacts/puffer_fixed_door_d1_v59_fresh_control_bc1m/flightrl_fixed_door_d1_seed11_1048576.bin \
  --training-report artifacts/puffer_fixed_door_d1_v59_fresh_control_bc1m/flightrl_fixed_door_d1_seed11_1048576.reevaluation.json \
  --prompt "interior door" \
  --threshold 0.25 \
  --duration-s 21 \
  --output artifacts/crazyflie_logs/door_puffer_shadow_v59_gateA_run2.csv
```

Only after the v59 reference is reviewed may T61 receive a separately named,
still non-actuating shadow:

```bash
uv run python scripts/crazyflie_door_puffer_shadow.py \
  --checkpoint artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.bin \
  --training-report artifacts/puffer_fixed_door_d1_t61_episode_rng_bc_seed11/flightrl_fixed_door_d1_seed11_1048576.promotion-seed20011.json \
  --prompt "interior door" \
  --threshold 0.25 \
  --duration-s 20 \
  --output artifacts/crazyflie_logs/door_puffer_shadow_t61_gateA_run1.csv
```

There is intentionally no live-control command in this handoff. Building a
readiness artifact would still return false because T61 fails the registered
8 deg/s simulation thresholds of 70% mission, 65% outside-FOV, and at most 3%
collision. Forward authority is additionally ruled out by the obstacle result.

Stop immediately and preserve the evidence if an output path already exists;
fewer than 50 synchronized rows exist; sampled coverage is under 20 s; any
checkpoint/report/contract/detector/device/hardware identity or SHA differs; a
row is nonfinite or actuating; timestamps or frame indices do not increase;
grounding results refer to a future/out-of-order frame; more than five frames
drop; result cadence is below 1 Hz; grounding p95 exceeds 750 ms or age margin
p05 is below 0.25 s; policy p95 exceeds 10 ms; frames are below 128x96 or not
near 4:3; search plus target/recovery phase coverage is missing; fewer than 20
detections or 10 strict alignment samples exist; any off-center detection has
zero/wrong-sign yaw; projected translation or executed previous action is
nonzero; or projected absolute yaw exceeds 8 deg/s or differs from the bound
action-contract mapping. A good trace permits review only. It cannot override
the failed simulation cap.

## Thread closeout and path to a working live policy

### What this thread established

- v59 is the frozen operational reference, but its native policy actually used
  the 4 rad/s physics scale while configuration and teacher semantics said
  70 deg/s. Its 79.11% result therefore cannot validate the intended live
  mapping.
- T61 is the strongest simulation policy: fresh-controller recurrent pure BC,
  corrected 70 deg/s native mapping, episode-indexed randomness, and
  1,048,576 transitions. It beats the matched v59 bundle by 14.74 mission
  points with lower collision and a 22.95-point worst-layout gain.
- Corrective DAgger did not explain the remaining failures. Its matched
  seed-11 result was 1.65 mission points worse than pure BC, so expanding it
  to seeds 23 and 47 would not add useful causal evidence.
- T61 does not satisfy the live yaw envelope. Post-hoc 8 deg/s clipping falls
  to 57.62% mission and 12.10% outside-FOV success. This is now the decisive
  blocker for yaw-only authority.
- Forward authority is separately blocked: the single obstacle intervention
  produced 3.33% mission and 30.07% collision.
- Recurrence matters, but the registered temporal-order scramble was nearly
  neutral. A World Action Model is therefore not the next justified
  intervention.
- Action, observation/recurrence, live safety, readiness, detector, episode
  stream, and evidence-age meanings now have distinct versioned owners. Native
  mappings and detector bytes are derived from those owners and fail closed.
- No physical drone or radio path was used. No real v59 or T61 shadow evidence
  exists yet.

### Definition of the first working live policy

The first defensible demo is intentionally narrow: firmware performs takeoff,
stabilization, height/position hold, landing, and abort; the learned recurrent
policy consumes the real camera stream and may command yaw only, capped at
8 deg/s, with every translation component hard-zero. Starting with the door
outside the camera FOV, it must find the door, turn with the correct sign,
center it, and hold alignment without a safety or evidence-gate violation.
Flying toward or through the door is a later milestone.

### Ordered continuation ladder

1. **Checkpoint the source before collecting promotion evidence.** Review the
   shared dirty worktree, form logical commits, and record the exact Git commit,
   `uv.lock` digest, policy/report hashes, detector-v3 SHA, hardware-config SHA,
   and firmware identity. Until then, a real trace is diagnostic rather than
   fully reproducible promotion evidence.
2. **Collect the non-actuating v59 reference.** Run the exact first command
   above, preserve its CSV/summary unchanged, and apply every registered stop
   condition. If coverage is short, use the separately named 21-second retry;
   never overwrite run 1.
3. **Collect a separately named T61 shadow only after v59 passes review.**
   Compare v59 and T61 on matched target placements. Require strict off-center
   yaw sign, nonzero yaw when alignment is needed, zero translation and
   previous action, detector-v3 identity, acceptable latency/age, and stable
   recurrent outputs. Repeat lighting, initial target visibility, and camera
   latency as separate traces rather than combining conditions.
4. **Test whether the 8 deg/s task is feasible before training again.** Run the
   privileged teacher/oracle under the same 8 deg/s native action contract and
   unchanged episode horizon. If the teacher misses 70% mission or 65%
   outside-FOV success, first change only the time horizon; do not blame the
   student for an infeasible task.
5. **Train T62 natively at the live yaw scale.** If the teacher is feasible,
   train a fresh recurrent BC policy whose declared and executed policy yaw
   are both 8 deg/s. Keep T61's budget, episode stream, observation contract,
   controller initialization, teacher, and evaluation fixed. Screen seed 11;
   evaluate seeds 23 and 47 only after it passes. Do not train at 70 deg/s and
   clip after the fact.
6. **Replay real shadow observations offline.** Feed the immutable v59/T61
   traces through T62 with recurrence and evidence age intact. Separate
   detector/calibration failures from controller failures. If real perception
   is the blocker, change one lighting, noise, latency, or detector variable
   at a time and re-run shadow; do not mix this with policy retraining.
7. **Build a new readiness artifact only after both gates pass.** Required
   inputs are a promotion-grade T62 simulation report at native 8 deg/s and
   matching real shadow evidence. The effective authority must serialize as
   `yaw_only`; translation must remain false. A passing shadow alone is not
   sufficient.
8. **Run the first supervised yaw-only trial.** This requires a reviewed
   operator procedure, firmware stabilization, a 10-15 second bound, 8 deg/s
   maximum yaw, translation hard-zero, immediate abort/land, and the rollback
   checkpoint above. Review the trace before any repeat or scope expansion.
9. **Treat bounded forward as a separate project gate.** First repair the
   obstacle failure with a separately evaluated safety mechanism or training
   intervention. Require collision at most 3% across matched obstacle seeds
   and repeated successful yaw-only trials before exposing any forward
   authority.

### Recommended engineering direction

Prioritize T62 at a native 8 deg/s action contract plus real-shadow replay.
That directly attacks the measured live blocker while preserving the small,
recurrent PufferLib actor and firmware boundary. Defer PPO, full World Action
Models, raw motor control, on-edge inference, and forward navigation until
evidence identifies one of them as the missing mechanism.
