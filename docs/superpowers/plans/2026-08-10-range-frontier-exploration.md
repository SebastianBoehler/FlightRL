# Range-Frontier Exploration v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify the single-drone range-map exploration policy spine through a non-actuating replay shadow artifact.

**Architecture:** A sparse takeoff-relative occupancy map exposes a 4x32x32 egocentric exploration crop. A Gymnasium environment and batched stepping core use the same mapper/frontier semantics as a gated per-frame actor-critic with no carried hidden state; PPO owns speed and direction while a separate shield owns safety vetoes. The first hardware-facing output is replay/shadow only.

**Tech Stack:** Python 3.13, NumPy, Gymnasium, PyTorch, existing FlightRL PufferLib 4 export/runtime patterns, pytest.

## Global Constraints

- Contract ID is `range-frontier-exploration-v2`; observation is exactly 4,106 float32 values.
- Actor controls only normalized forward `[0, 1]` and yaw `[-1, 1]`; altitude and safety remain outside the actor.
- Simulator truth may affect reward and reports but never actor inputs.
- Files remain below the 300 LOC soft limit; no camera/swarm/radar implementation.
- No checkpoint gains shadow, deployment, or flight authority from training alone.
- Preserve unrelated worktree changes and `uv.lock`.
- Do not commit unless the user explicitly asks; use verified diffs as checkpoints.

## Current Status (2026-08-10)

Tasks 1--6 are implemented and verified. Candidate `v14_gated_map_memory_full`
passes the 1,200-step, eight-seed simulation gate and both retained calibration
flights pass replay shadow v2 with `eligible_for_live_shadow=true` and
`controls_drone=false`. Task 7 remains deferred; native PufferLib export is not
required for the next host-side non-actuating live shadow.

---

### Task 1: Versioned contract and observation builder

**Files:**
- Create: `src/flightrl/exploration/range_contract.py`
- Create: `src/flightrl/exploration/range_observation.py`
- Test: `tests/test_range_exploration_contract.py`

**Interfaces:**
- Produces: `range_exploration_contract_payload() -> dict[str, Any]`
- Produces: `build_range_exploration_observation(map_crop, ranges, validity, previous_action) -> np.ndarray`
- Produces constants `RANGE_EXPLORATION_OBSERVATION_DIM`, `RANGE_MAP_SHAPE`, and `RANGE_ACTION_DIM`.

- [ ] Write a failing test asserting literal segment offsets, 4,106 values, dtype, finite/range validation, prohibited privileged inputs, and all authority flags false.
- [ ] Run `pytest -q tests/test_range_exploration_contract.py` and confirm import failure.
- [ ] Implement the immutable payload and strict flattening builder; reject wrong shapes, non-finite values, non-binary validity, or out-of-range actions.
- [ ] Re-run the focused test and `git diff --check`.

### Task 2: Sparse mapper and all-frontier extraction

**Files:**
- Create: `src/flightrl/exploration/range_mapper.py`
- Test: `tests/test_range_exploration_mapper.py`

**Interfaces:**
- Produces: `RangePose(x_m, y_m, yaw_rad)` and `RangeOccupancyMap`.
- `RangeOccupancyMap.update(pose, ranges_m, validity, *, visited=True) -> None` applies the exact log-odds ray contract.
- `RangeOccupancyMap.exploration_crop(pose) -> np.ndarray` returns `(4, 32, 32)` visited/free/occupied/reachable-frontier data.
- `RangeOccupancyMap.frontier_cells(pose) -> set[tuple[int, int]]` returns all reachable clusters of at least three cells.

- [ ] Write failing hand-derived tests for one finite ray, no-return clearing without an endpoint, repeated occupied/free thresholds, frontier cluster filtering, body-frame crop rotation, and reset.
- [ ] Run `pytest -q tests/test_range_exploration_mapper.py` and confirm import failure.
- [ ] Implement grid indexing, Bresenham-style ray cells, log-odds updates, reachable-free flood fill, cluster extraction, and nearest-neighbor egocentric crop sampling.
- [ ] Re-run mapper and existing `tests/test_ranger_map.py tests/test_spatial_memory.py`; check whitespace.

### Task 3: Gymnasium environment and batched parity

**Files:**
- Create: `src/flightrl/exploration/range_world.py`
- Create: `src/flightrl/exploration/range_env.py`
- Test: `tests/test_range_exploration_env.py`

**Interfaces:**
- Produces: `RangeWorld.generate(seed) -> RangeWorld`, with a connected 64x64 truth grid at 0.10 m/cell.
- Produces: `RangeExplorationEnv(seed, maximum_episode_steps=1200, stress=False)`.
- Produces: `range_batch.RangeExplorationBatch(num_envs, seed, maximum_episode_steps, stress)` with `observations`, `step(actions)`, and `reset_done(mask)` arrays.

- [ ] Write failing tests for deterministic connected worlds, literal ray ranges, Gymnasium checker, reset/step shapes, collision=-2 termination, total positive reward <=1, and single/batch parity for fixed actions.
- [ ] Run `pytest -q tests/test_range_exploration_env.py` and confirm import failure.
- [ ] Implement connected room/obstacle generation, disc collision, 20 Hz kinematics, four ray sensors, truth coverage accounting, and the shared observation path.
- [ ] Add the exact stress envelope: fixed per-episode range bias, dropout bursts, odometry scale/yaw drift, and 0/2/5-step action lag selected from the seeded RNG.
- [ ] Re-run the environment test plus `tests/test_coverage_exploration.py`; check whitespace.

### Task 4: Gated map-memory policy and PPO update

**Files:**
- Create: `src/flightrl/exploration/range_policy.py`
- Create: `src/flightrl/exploration/range_ppo.py`
- Test: `tests/test_range_exploration_policy.py`
- Test: `tests/test_range_exploration_ppo.py`

**Interfaces:**
- Produces: `RangeExplorationActorCritic(hidden_size=64)` below 100,000 parameters.
- `forward_step(observation) -> (location, value)` with no external neural state.
- Produces: `RangePpoConfig`, `collect_range_rollout`, and `range_ppo_update`.

- [x] Write failing policy tests for output shapes, parameter count, finite validation, and the absence of external recurrent state.
- [x] Implement the four-channel CNN, scalar MLP, zero-state gated encoder, bounded actor head, and critic.
- [ ] Write a failing PPO smoke test proving one real rollout/update changes parameters and returns finite losses without exposing truth fields.
- [x] Implement bounded action sampling, GAE, clipped PPO/value/entropy losses, and gradient clipping.
- [ ] Re-run policy/PPO tests and the existing bounded-action tests; check whitespace.

### Task 5: Baselines, causal evaluation, and checkpoint gate

**Files:**
- Create: `src/flightrl/exploration/range_evaluation.py`
- Create: `src/flightrl/exploration/range_checkpoint.py`
- Create: `scripts/train_range_exploration.py`
- Create: `scripts/evaluate_range_exploration.py`
- Test: `tests/test_range_exploration_evaluation.py`
- Test: `tests/test_range_exploration_checkpoint.py`

**Interfaces:**
- Produces stationary-scan and deterministic classical-frontier baseline actions.
- Produces `evaluate_range_candidate(...) -> dict` with clean, range-masked, map-masked, obstacle-counterfactual, stress, and baseline metrics; mirrored-frontier behavior is diagnostic only.
- Produces hash-bound checkpoint save/load; loader rejects missing/contradictory authority, contract, seed, metric, or state evidence.

- [ ] Write failing tests proving the classical controller selects a direction but never enters actor observation, and mirrored maps require opposite target yaw.
- [ ] Implement deterministic baselines and fixed held-out evaluation.
- [ ] Write failing forged-checkpoint tests for contract, state digest, evaluation scope, failed causal gate, and false authority claims.
- [ ] Implement checkpoint binding and CLIs with unique output paths and no W&B dependency.
- [ ] Run a bounded local PPO smoke, save its honest report even if gates fail, and re-run the focused evaluation/checkpoint tests.

### Task 6: Non-actuating telemetry replay shadow

**Files:**
- Create: `src/flightrl/exploration/range_shadow.py`
- Create: `scripts/replay_range_exploration_shadow.py`
- Test: `tests/test_range_exploration_shadow.py`

**Interfaces:**
- `replay_range_shadow(checkpoint, telemetry_csv, output_dir) -> dict` consumes the exact current ranger telemetry schema.
- Logs every device timestamp, map reset/frontier count, raw action, shielded action, and safety reason. It sends no commander or cflib calls.

- [ ] Write a failing integration test using a literal telemetry CSV and passing simulation-only checkpoint; assert deterministic JSONL rows and `controls_drone=false`.
- [ ] Add failing tests for stale/out-of-order timestamps, missing columns, low Flow quality, invalid range, non-finite output, and checkpoint gate failure.
- [ ] Implement replay, the pure safety shield, manifest hashes, bounded output creation, and CLI.
- [ ] Replay both existing passing calibration flights and retain reports under their run directories without promoting authority.
- [ ] Run the complete new range slice plus current flight telemetry/validation tests and `git diff --check`.

### Task 7: PufferLib 4 integration boundary

**Files:**
- Create: `src/flightrl/exploration/range_puffer_export.py`
- Create: `scripts/export_range_exploration_puffer4.py`
- Test: `tests/test_range_exploration_puffer4.py`

**Interfaces:**
- Exports a self-contained `flightrl_range_exploration_v1` Ocean source/config bundle to a caller-supplied clean PufferLib 4 checkout.
- Uses `resolve_pufferlib_root` and source hashing; it refuses a dirty dependency checkout and never edits the current known-dirty checkout.

- [ ] Write failing export tests against a temporary minimal PufferLib fixture and a rejection test for a dirty/non-checkout target.
- [ ] Implement exact contract/config/source export and a dry-run manifest without building.
- [ ] Verify exported observation/action sizes and checkpoint contract match the Python lane byte-for-byte.
- [ ] Do not build or mutate `/Users/sebastianboehler/Documents/GitHub/PufferLib-4-flightrl` while it contains unrelated changes.

## Final verification

- [ ] Run all new range exploration tests plus the adjacent existing coverage, ranger-map, flight-telemetry, flight-validation, and script suites.
- [ ] Run Gymnasium checker, a deterministic training smoke, both live-log replay shadows, and `git diff --check`.
- [ ] Report exact passing/failing gates. Request a live scripted shadow flight only if the offline checkpoint and replay contract permit it; otherwise state the next failing offline gate.
