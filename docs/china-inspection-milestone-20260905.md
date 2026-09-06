# Three-panel inspection geometry checkpoint

Implemented directly in the original FlightRL `main` checkout following approval.
The previous native diagnostic replay checkpoint was transferred from the retained
worktree. Existing unrelated edits were preserved. No commit or push performed.

## Implemented boundary

An owned authored room contains three solid red/green/blue square markers and one
box occluder. The scenario v1 compiler supplies the same room/obstacle arrays to
native camera ray casting and swept collision detection. A separately hashed
inspection scene adds rectangle transforms, RGB appearance and evaluator-only IDs.
The panel surface normal is `u cross v`; local coordinates use metres, FLU body
axes and world-from-body wxyz quaternions. Camera offset, field of view and
projection are explicitly recorded. Static geometry is shared across the batch.

The new C renderer writes preallocated RGB frames and evaluator-only visible /
unoccluded projected pixel counts. Camera and collision calls are batched; Python
iterates over ticks, with observation bookkeeping outside the simulation loop.
This is a CPU reference backend, not Metal or a training-throughput claim.

Collision uses a swept axis-aligned cube with 0.08 m half extents. This conservative
body envelope detects contact, including crossing a thin obstacle in one tick.
Contact terminates the diagnostic batch and retains its last valid state, while
recording attempted positions and per-environment collision flags. It does not
simulate contact forces. A non-colliding peer receives a batch-stop event. Panel
rectangles are appearance surfaces, not additional solid volumes; the authored
fixture mounts them immediately in front of the room wall.

## Observation and evaluation separation

`detect_markers` accepts only actual RGB frames. It recognizes the explicitly
assumed unique solid color markers; this is not general industrial panel recognition.
Memory associates visible color identities and records discovery / first useful
observed view. Repeated useful views increment a duplicate counter without adding
unique inspection credit. No target coordinates, evaluator IDs, scene occupancy,
perfect pose or panel-count oracle reach that memory.

Observation-side useful-view checks use pixel area, bounding-box fill/aspect ratio
and image-boundary clearance. These are proxies. Independent truth evaluation uses
at least 64 visible pixels, 95% visibility, distance at most 3 m, facing cosine at
least 0.75, and all panel corners inside the camera field of view. The camera is
instantaneous and ideal: blur, sensor noise and real inspection-defect resolution
are not modeled or validated. Both observed and evaluator results are retained;
an observation proxy is not silently promoted to ground-truth quality.

The actor ends at budget exhaustion with coverage unknown. Evaluator results list
missed IDs and whether the entire authored panel set received useful views. This
fixture's panel set is accessible, but no general discoverability solver exists.
No estimator, exploration planner, learned policy or recovery is implemented.

## Replay and reproduction

`flightrl.inspection_replay.v2` explicitly stores RGB frames, states labeled truth,
actions, link observations, panel counts labeled truth, attempted positions,
collision flags and timestamp-indexed mission/link events. Scene, native binary,
recorder and file identities are bound in manifests. The original gray4 replay v1
remains supported unchanged. Digests detect accidental alteration, not trusted
signer provenance. Operator frame access returns no frame during scripted dropout;
local recording continues. Dropout is not an RF model or a recovery demonstration.

From this checkout:

```sh
PYTHONPATH=src .venv/bin/python setup.py build_ext --inplace
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_inspection_scene.py tests/test_inspection_replay.py tests/test_scenario_replay.py tests/test_scenario_bundle.py tests/test_native_core_contract.py tests/test_native_sixdof_vision.py
PYTHONPATH=src .venv/bin/python scripts/capture_inspection_room.py --output artifacts/inspection-new-run
```

Use a new output path; existing artifacts are never overwritten. The script reloads
and compares the scene/replay and emits native camera PPMs plus a JSON report.
Three fixed-start hover diagnostics produce 303 frames, with 120 operator frames
withheld. Their combined viewpoints validate panels 101 and 103; panel 102 remains
uninspected. This union is diagnostic coverage across separate runs, not one drone
completing a mission. Each run correctly reports incomplete coverage.

Tests cover independently calculated projected pixel counts, behind-camera views,
partial/full occlusion, oblique/clipped/far views, target removal, changed hidden
IDs, repeated observations, budget termination, swept contact, replay corruption
and native shape rejection. No browser, Metal, learning or physical flight test.

## Next milestone

Build the same-information classical exploration baseline and define the estimator
interface, then a minimal selected-episode viewer. Measure Metal/learner exchange
before a large port. Learning, route memory/recovery, held-out evaluation and final
presentation remain later milestones. Full inspection mission scope is retained.
