# FlightRL review and remediation — 6 September 2026

Reviewed the dirty FlightRL checkout at HEAD `5d20e4e`, including untracked
executable files. Findings were reported before edits. No commit, push,
hardware run, model retraining or paid compute was performed. Existing
experiment evidence remains intact.

## Confirmed defects addressed

| Priority | Defect | Resolution |
| --- | --- | --- |
| P1 | First RGB-D depth frame aliases the capture buffer reused across drones | Own the initial depth array, as later frames already do |
| P1 | Review packaging rewrites old evaluations with current source hashes | Write a separate input-artifact manifest; never rewrite evaluation inputs |
| P2 | Environment scenes fail storage round-trip | Restore the environment and compare validated canonical identities |
| P2 | Confirmer accepts an occluded or distant target | Apply distance and line-of-sight checks after the delayed-report gate |
| P2 | Eight-metre no-hit returns become obstacles | Preserve observed free space without marking a fictitious surface |
| P2 | Mission termination uses the preceding capture timestamp | Count executed physics steps; retain last capture time and terminal position separately |
| P2 | Async episode/image loads can complete out of order | Commit a fully decoded episode only when it is the latest request |
| P2 | Scene replacement retains GPU resources | Dispose shared mesh resources, textures, instances and shadow targets |
| P2 | Adaptation reruns overwrite the retained initial actor | Refuse an existing adaptation before checkpoint writes or training |
| P2 | Learned trial claims an unchecked upstream/model revision | Verify clean imported Git revision and pinned checkpoint digest before inference |
| P2 | Degenerate monocular motion aborts evaluation | Report unavailable metrics and the reason, retaining tracking availability |
| P2 | Wrapped reconstruction link is hidden beneath the fixed-height header | Let the header grow with wrapped navigation; verify narrow-screen navigation |

The final row was discovered during browser verification after the read-only
phase. The other eleven findings came from that phase. Camera actor inputs
remain RGB-D, body-frame proprioception, roles and delayed reports. No simulator
pose, world target or evaluator alignment was added to inference.

## Performance measured on the local M4 Max

Native extension rebuilt locally. These are workload-specific measurements,
not hardware transfer or training throughput claims.

| Workload | Before | After | Measurement |
| --- | ---: | ---: | --- |
| Seed 4100 reconstruction, 684 camera frames | 146.4 fps | 183.1 fps | Median of three runs per version; native capture, both VO backends and fusion |
| Seed 4100 whole experiment | 5.68 s | 4.79 s | Also includes flight, final scoring and in-memory exports; excludes file/browser I/O |
| Fusion-only code substitution within the corrected pipeline | 139.9 fps | 163.6 fps | Three alternating pairs; identical complete map/pose/metric outputs |
| Utility plant native capture, 256×192, batch 1 | 127.8 fps | 171.9 fps | Three timing blocks of 15 captures |
| Data centre native capture, 256×192, batch 1 | 106.4 fps | 122.0 fps | Same protocol |
| Forest native capture, 256×192, batch 1 | 117.8 fps | 115.7 fps | No demonstrated gain at this workload |

Fusion batches finite-point filtering and voxel coordinates while retaining
first-observation ordering and timestamps. Native shading skips zero-power
contributions. RGB, ray distance and evaluator counts matched byte-for-byte in
12 scene/batch/resolution workloads. The initial profile attributed 2.41 s to
native capture, 1.24 s to fusion, 1.27 s to VO and 0.67 s to final scoring in a
6.05 s profiled run; overlapping cumulative timings are not additive.

Viewer trajectories now allocate once per dataset and change their draw range
on replay. Gaps in tracking stay gaps. The unused `LatestFrame` implementation
and its two tests were removed after confirming there were no runtime consumers.
A resource-lifecycle regression covers the active disposal path instead.

## Corrected held-out results

Seeds 4100–4105, frozen actor, unchanged scenes and thresholds: **0/6 complete,
4 collisions, 2 incorrect reports**. Aggregate reconstruction throughput was
180.6 camera frames/s. This rate covers the same capture/VO/fusion stages above.

| Backend | Mean tracking availability | Median tracked-fragment ATE | Median surface error | Pass tracking ≥95% and ATE ≤0.15 m |
| --- | ---: | ---: | ---: | ---: |
| RGB | 13.7% | 0.014 m | 1.276 m | 0/18 |
| RGB-D | 96.5% | 0.701 m | 0.276 m | 1/18 |

The depth ownership bug slightly changes RGB-D error, but does not explain the
larger drift. Low monocular ATE over short surviving fragments is not success.
Surface coverage remains restricted to surfaces seen along the flight, not the
whole site. The mapper still lacks landmark replenishment, loop closure,
relocalization and an uncertainty-aware planning map. Detailed forest replay
remains visually different from the actor's native observations.

## Research and dependency check

- LingBot's imported checkout and live GitHub HEAD both resolve to `bfcd0f20383d3a35cc9757a36ab1d5b6e5064df4` (31 August). The trial now verifies this code and the existing checkpoint SHA-256 `ee665103348e07e6b826d529b8e61de8f413d5432a4f2e84970d6c8fd2e1cd72`, downloaded from HF revision `204754b72bb24f561f8d7e7e1e4e4cd9e809adf9`. No dependency replacement is justified by the current results. [Official implementation](https://github.com/Robbyant/lingbot-map)
- Failure availability and drift must be measured separately. Corruption fidelity can reverse a tracker ranking; visually plausible imagery alone is insufficient validation. [Failure or Drift?, 31 August](https://arxiv.org/abs/2608.30690)
- Local inertial tracking plus overlap-conditioned keyframes and central submap alignment is a relevant design reference. It is not an implemented FlightRL capability. [CGS-SLAM, 27 August](https://arxiv.org/abs/2608.26868)
- Cross-agent pointmap verification, similarity-gauge synchronization and global optimization address capabilities missing from the current one-shot registration baseline. Its reported 8 fps cannot be transferred to this Mac or workload. [CoMo3R-SLAM](https://arxiv.org/abs/2605.30488)
- VGGT-SLAM 2.0 supplies a loop-closure/submap baseline and a real-time entry point; it would require a separate measured integration, not a renamed existing mapper. [Official implementation](https://github.com/MIT-SPARK/VGGT-SLAM)

Engineering priority: maintain reliable local tracking first; assess learned
geometry asynchronously; accept shared-map constraints only on validated
overlap. Keep physical free-space/unknown-space reasoning distinct from a
renderable surface cloud. These are proposed research steps, not added features.

## Validation and retained evidence

- 75 focused Python tests passed, covering actor isolation, geometry, replay
  identity, cooperative protocol, native/Metal parity, contacts and environments.
- Viewer typecheck/build passed; existing large-bundle warnings remain.
- Active GPU resource-disposal regression passed in Node.
- Actual recorded clips were packaged; all source JSON/NPZ hashes were unchanged.
- Actual adaptation rerun was refused; all checkpoint hashes were unchanged.
- Actual learned trial passed code/checkpoint verification and then refused the
  existing output directory. Learned inference was not rerun.
- Browser checked scene switching, rapid selection, end/rewind, map/drone
  selection, initialization hiding and responsive navigation. Resource release
  was checked by disposal events, not a long-duration GPU memory soak.

Local evidence: `artifacts/review-remediation-20260906/` contains the source
snapshot, timings, parity captures, guard logs, copied prior public review,
packaged recordings and fresh held-out results. The default reconstruction
viewer now shows the corrected seed-4100 export. Original learned-trial
artifacts and earlier failed/repaired experiments were retained.

Simplify & Harden completed in 41 seconds: two cosmetic changes and one metric
hardening fix, 17 additional changed lines against a conservative 388-line
implementation diff. Stationary reference motion now also reports unavailable
monocular scale instead of permitting zero-scale alignment. The focused nine
reconstruction tests passed again. No structural refactor was proposed.
Structured output: `artifacts/review-remediation-20260906/simplify-and-harden.yaml`.
