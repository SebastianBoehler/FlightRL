# Camera reconstruction: research and implementation, 6 September 2026

The implementation target is an incrementally reconstructed site from onboard observations, with no simulator poses or obstacle map supplied to the reconstruction backend. The learned flight actor already consumes RGB-D, ideal body odometry/IMU, roles and delayed reports. Reconstruction and flight control are separate components; a reconstruction score does not prove navigation safety.

## Recent primary sources

| Work | Date | Useful direction | Integration decision |
|---|---|---|---|
| [CGS-SLAM](https://arxiv.org/abs/2608.26868) | 27 Aug 2026 | Collaborative RGB/inertial tracking, monocular depth priors, keyframe exchange and centralized reconstruction | Strong architectural reference for local tracking plus asynchronous shared mapping; not a validated dependency here |
| [Failure or Drift?](https://arxiv.org/abs/2608.30690) | 31 Aug 2026 | Distinguishes tracking failure from silently inaccurate tracking under corruptions | Report both tracking availability and pose error; synthetic lighting changes cannot establish real-world robustness |
| [CoMo3R-SLAM](https://arxiv.org/abs/2605.30488) | 28 May 2026 | Collaborative reconstruction priors, submap matching, similarity synchronization and global optimization | Reference for independently initialized cameras and map alignment; never merge using hidden simulator spawn poses |
| [LingBot-Map](https://arxiv.org/abs/2604.14141), [code](https://github.com/Robbyant/lingbot-map) | 15 Apr 2026 | Streaming geometric reconstruction with bounded context | Published weights evaluated locally on one development sequence using the SDPA path on Apple MPS; see measured trial below |
| [VGGT-SLAM](https://github.com/MIT-SPARK/VGGT-SLAM), [author update](https://gtsam.org/2026/06/24/vggt-slam.html) | 24 Jun 2026 update | Overlapping submaps, loop closures and projective alignment | Candidate persistent mapping backend; projective ambiguity can exceed simple scale alignment |

These papers solve different tasks and use different sensors, hardware and benchmarks. Their reported speeds and accuracy are not our measurements, and do not establish a single universal state of the art. Repository availability does not by itself verify pretrained-weight licensing or deployment compatibility.

## Implemented baseline

`flightrl.reconstruction` implements calibrated optical geometry, incremental RGB-D PnP odometry, a monocular essential-matrix/triangulation/PnP baseline, voxel-sampled colored surface fusion, overlap-verified RGB-D submap registration, and an evaluator isolated from the mapper. OpenCV is pinned in the optional `reconstruction` extra.

All maps start empty. RGB-only odometry receives no depth, IMU, known extrinsics or world positions. It has arbitrary scale; a similarity alignment is used **only by the evaluator**. RGB-D uses ray-distance depth, not camera-axis Z depth. Its only coordinate gauge is the first camera frame. Simulator poses are used by the image renderer and evaluator, never by odometry or fusion.

This is visual odometry and surface reconstruction, **not complete SLAM**: no loop closure, bundle adjustment, new monocular landmark insertion, dynamic-object rejection, occupancy/free-space inference or active exploration is implemented. Monocular landmark depletion and texture-poor walls are explicit failure cases. Tracking stops rather than substituting true poses. A colored surface cloud is not a collision-safe map.

The frozen learned actor flies three beacon roles in previously untrained equipment/lighting variants. Reconstruction consumes a co-located 256 × 192 simulated camera at 10 Hz; the actor retains its original 64 × 48 RGB-D stream. Reconstruction runs offline over the actual closed-loop trajectory. This is mapping during flight, not a map-driven exploration policy or a reconstruction model trained in this turn.

## Evaluation and next backend

Report camera tracking fraction, absolute trajectory RMSE, endpoint error, consecutive-frame translation error, nearest-surface mean/P95 error and coverage within 15 cm. Coverage is over surfaces actually visible along the trajectory; it is **not whole-site completion**. Retain mission collisions, failed reports and timeouts alongside completion times. Ground truth is retained only in the evaluator and optional review overlay.

The next backend experiment should compare a released learned streaming reconstruction model against this baseline on identical camera sequences, including turns, revisits, weak texture, motion blur and moving objects. Add loop closure and bounded local free-space mapping before making reconstructed geometry an input to exploration. Multi-drone fusion must reject unverified overlap and measure registration error; delayed keyframes and uncertainty belong in peer messages, not raw ground-truth maps. Real metric monocular operation needs a scale source such as calibrated inertial motion, a known dimension or an explicitly evaluated learned prior.

## Measured outcome

Artifacts: `artifacts/reconstruction-20260906/heldout-repaired/`. Six held-out seeds 4100–4105, three cameras per seed, frozen actor checkpoint. The first attempt is retained in `heldout/`; it stopped at a rounded pixel index at the image boundary. The repair clips sample coordinates to valid pixel indices and reruns the same declared seeds. It does not retune the actor or odometry against their measured errors.

| Measurement | RGB only | RGB-D |
|---|---:|---:|
| Mean fraction of tracked frames, across 18 sequences | 13.7% | 96.5% |
| Median trajectory RMSE, tracked frames only | 0.014 m after evaluator Sim(3) alignment | 0.708 m, first-frame gauge only |
| Median mean surface error | 1.276 m | 0.276 m |

The small monocular trajectory error is misleading in isolation: most motion was not tracked, and similarity alignment was fitted to a short surviving fragment. This baseline does **not** demonstrate robust monocular reconstruction. RGB-D also fails the intended navigation accuracy despite high tracking availability. The predeclared 95% tracking / 15 cm trajectory-error checks are necessary checks, not certification of map or flight safety.

Flight outcomes: **0/6 complete, 4 collisions, 2 incorrect reports**, last-capture times 8.8–22.7 s. These are cooperative beacon missions in changed equipment/appearance, not complete search-and-rescue or site exploration. No mapping completion time can be claimed. Only 1/12 attempted cross-drone registrations was accepted; evaluator translation error was 0.331 m and rotation error 3.29 degrees. The review deliberately retains independent maps rather than claiming reliable merged swarm reconstruction.

Measured throughput: **135 camera frames/s** across the six runs, each frame processed by both RGB and RGB-D baselines. Timed section includes 256 × 192 native rendering, both trackers and surface accumulation; excludes final nearest-neighbor scoring, mission rollout, JPEG/JSON/PLY writing and browser rendering. This is CPU reconstruction throughput, not training steps/s or a learned-model benchmark.

Review: `/mapping.html`, linked from the normal dashboard. It shows an empty initial map, capture-synchronized camera replay, incremental surfaces, independent drone/backend selection, reference-trajectory toggle, tracking-loss status and downloadable PLY maps. Seven regression tests plus three actor-boundary tests pass; TypeScript/Vite build passes. Browser checks cover empty state, advancing camera timestamps and point counts, backend/drone selection, tracking-loss display and reference overlay.


## Published learned model: actual local trial

The released LingBot-Map checkpoint was run on RGB alone: no simulator depth, calibration, poses or map entered inference. Upstream revision `bfcd0f20383d3a35cc9757a36ab1d5b6e5064df4`, checkpoint SHA256 `ee665103348e07e6b826d529b8e61de8f413d5432a4f2e84970d6c8fd2e1cd72`. The strict checkpoint load used all published weights. An immutable rotary cache was converted from complex128 to complex64 for MPS compatibility; a small upstream CPU/MPS rotary check matched exactly. No trained weights were changed.

This is **one development trajectory**, seed 4000, drone 1: 19 RGB keyframes sampled at 1 Hz, resized to 266 × 196. Four-frame initialization means the first reconstruction becomes available at capture time 3 s. Float32 SDPA inference on Apple M4 Max took **4.34 s / 4.38 frames per second**, including initialization, excluding model loading, rendering and evaluation. It is offline mapping over a recorded closed-loop flight, not online map-conditioned control or a held-out learned-model benchmark.

| Measurement | Corrected learned RGB result |
|---|---:|
| Offline predicted poses | 19 / 19 (not independently certified tracking) |
| Trajectory RMSE after evaluator-only Sim(3) | 0.127 m |
| Endpoint error after the same alignment | 0.094 m |
| Consecutive translation RMSE | 0.102 m |
| Mean / P95 nearest-surface error | 0.399 / 0.846 m |
| Observed reference coverage within 15 cm | 0.35% |
| Exported confidence-filtered points | 2,038 |
| Flight outcome | Collision at 18.0 s |

The cloud uses pixel stride 8 and the author's confidence threshold 1.5. Sparse sampling and depth/pose errors affect coverage; this is not a dense or complete digital twin. Metric scale was fitted only for evaluation and is not available to a deployed monocular mapper. The result cannot be directly ranked against the six-seed classical table: different sequences, frame rates and processing are involved.

A pose-convention defect was caught before the final result: an upstream utility docstring describes world-to-camera poses, but the official benchmark explicitly consumes this checkpoint's decoded poses as camera-to-world. The corrected export follows the benchmark and backprojects predicted camera-Z depth using predicted intrinsics. It does not use the simulator's ray-distance convention. An analytic rotated/translated-camera regression protects this distinction. Invalid preliminary exports remain under `learned-trial-inverted-export/` for audit; only `learned-trial/` is the final result.

Review: `/mapping.html?backend=lingbot`. Reproduction: install the optional reconstruction dependencies plus einops, obtain the pinned upstream source and published checkpoint, run `scripts/trial_lingbot_reconstruction.py`, then `scripts/evaluate_learned_reconstruction.py`. The trial refuses to overwrite its output directory. Full configuration, inputs, predictions and measurements are retained in `artifacts/reconstruction-20260906/learned-trial/`.

The recommended next implementation remains local visual-inertial tracking with asynchronous learned geometry, followed by loop closure, uncertainty-aware occupancy/free-space mapping and overlap-validated shared submaps. Active exploration must then use that observed map, with new held-out flights scored for collisions and completion. These components remain unfinished; the current result establishes a measured reconstruction backend and review surface, not safe autonomous mapping of arbitrary sites.
