# China software demonstration: implementation and review

All work is local in the original FlightRL `main` checkout. No commits, pushes,
paid compute/API calls, hardware purchases, physical flight or frozen-hackathon
writes. This is a bounded simulation demonstration, not production readiness.

## Implemented

1. **Observation-based baseline:** RGB-D camera observations build a local free /
   occupied map; grid planning approaches visually discovered markers. Explicit
   memory prevents repeat inspection credit. Native six-DoF dynamics execute the
   commands through the existing velocity/yaw controller. Hidden panel IDs and
   world geometry are never passed to the planner.
2. **Metal sensing:** custom camera kernel writes persistent PyTorch MPS-owned RGB
   and depth tensors. Geometry stays resident; pose uploads and synchronization are
   measured. CPU remains the single-episode runner because it is faster at batch 1.
3. **Learning:** a 22,656-parameter CNN/GRU local controller trained from 5,779
   classical demonstrations and 2,817 corrective student-state demonstrations.
   Three seeds, validation-only checkpoint selection, then one frozen checkpoint.
   Training uses MPS; selected-episode inference uses CPU. Mission planning,
   color-marker recognition, scan behavior, altitude limits and the local depth
   brake remain explicit and shared between methods. No claim that all mission
   logic is learned. PPO/differentiable alternatives were not implemented.
4. **Recovery:** link loss switches to reverse traversed-route memory. The camera
   map is checked again before moving. A spatial link model controls observed
   connectivity; returning changes the reconnection condition. No connectivity
   oracle is supplied. Blocked route, continued outage and estimator loss stop
   safely in the tested cases. Local images remain recorded.
5. **Evaluation and presentation:** 20 frozen test-layout variations, matched
   sensing/planning/safety budgets, full failure results, actual native replays,
   and a TypeScript/WebGPU viewer with camera pose/frustum/image plane and route.

## Results and limits

| Measure | Classical | Frozen learned controller |
|---|---:|---:|
| Mean panel coverage, 20 test layouts | 96.7% | 96.7% |
| Fully inspected test layouts | 18/20 | 18/20 |
| Collisions, inspection + recovery | 0/40 runs | 0/40 runs |
| Reconnection runs | 20/20 | 20/20 |
| Mean inspection duration including incomplete runs | 67.54 simulated s | 66.99 simulated s |

The aggregate promotion gate was fixed at >=90% mean coverage, zero observed
collisions and >=90% recovery. Both pass this **narrow simulation gate**. The
learned controller has not demonstrated a meaningful advantage over the baseline.
Small changes to one room family are not arbitrary-site or real-flight transfer.

In the original authored presentation room, classical inspection completes 3/3;
the learned replay stops with 2/3. It remains available in the viewer as an honest
failure case. Coverage is not silently called complete. The recovery replay ends
after reconnection and reports its remaining inspection work.

The sensor contract was explicitly stated during implementation: ideal simulated
RGB-D, known takeoff origin, ideal attitude, and integration of noisy/biased
world-frame velocity. It is a **modeled odometry interface**, not an implemented VIO
or validated inertial estimator. Camera measurements use ray-range depth in metres.
Color markers and nominal sizes are diagnostic assumptions. There is no real
camera noise/blur, changing illumination, semantic panel detector or arbitrary
object data association. The actor can exhaust its observed map while missing
panels. Its stopping state is not an evaluator completion oracle.

Native collision checks use a swept 0.08 m axis-aligned body envelope, not contact
physics. Dynamic barrier injection is recorded as a runtime event. Image quality
is the previously frozen ideal-camera visibility/distance/angle criterion.

## Performance and cost evidence

The completed demonstration/corrective training experiment took 68.3 wall seconds
on this M4 Max, including teacher collection and validation. It followed one failed
MPS pooling warm-up; that attempt's duration was not instrumented. The failure was
fixed by using a supported fixed pooling operation, without CPU fallback.
Evaluation duration is recorded separately in the final `evaluation.json`.
Authoring time, energy and hardware amortization were not measured; these numbers
are not complete service delivery cost or proof of an end-to-end speed advantage.

| Camera batch | Native CPU p50 | Metal pose upload/render/sync p50 |
|---|---:|---:|
| 1 | 0.037 ms | 0.387 ms |
| 32 | 1.421 ms | 0.362 ms |
| 64 | 2.843 ms | 0.849 ms |

CPU timings include evaluator pixel counts. Metal's sensor outputs do not include
those counts. RGB parity was exact over the benchmark batches; maximum depth
error was 2.9e-6 m. Metal rendering + input conversion + actual learner update
measured 4.18 ms p50 at batch 32. Data lifetime is explicit: two pose uploads,
resident RGB-D outputs, conversion/allocation for learner layout, and explicit
synchronization. Unified memory is not claimed to make the entire pipeline zero-copy.

The kernel uses the installed PyTorch MPS shader API, also documented in
[official PyTorch documentation](https://docs.pytorch.org/docs/2.8/generated/torch.mps.compile_shader.html).

## Artifacts and reproduction

- Training: `artifacts/china-demo-training-20260905-v2/training.json` and `selected.pt`.
- Final test results and browser data: `artifacts/china-demo-evaluation-20260905-final/`.
- GPU benchmark: `artifacts/china-demo-metal-20260905.json`.
- Screenshots/review: `artifacts/china-demo-review-20260905/`.
- Viewer instructions: `viewer/README.md`.

```sh
PYTHONPATH=src .venv/bin/python setup.py build_ext --inplace --force
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_inspection_autonomy.py tests/test_inspection_scene.py tests/test_inspection_replay.py tests/test_scenario_replay.py tests/test_scenario_bundle.py tests/test_native_core_contract.py tests/test_native_sixdof_vision.py tests/test_native_extension_sources.py
PYTHONPATH=src .venv/bin/python scripts/train_inspection_student.py --output artifacts/new-training
PYTHONPATH=src .venv/bin/python scripts/benchmark_inspection_metal.py --output artifacts/new-metal.json
PYTHONPATH=src .venv/bin/python scripts/evaluate_inspection_demo.py --checkpoint artifacts/new-training/selected.pt --output artifacts/new-evaluation
npm run build --prefix viewer
```

46 focused tests passed, including camera parity, observation/evaluator isolation,
closed-loop completion, reconnection and failure envelopes. Native output-buffer
validation was corrected and the extension forcibly rebuilt because setuptools
had not noticed an included-header change. Tests passed after rebuilding.

The final evaluator reran the same frozen checkpoint after the rebuild and adds
native binary, source and scene identities. The test set was not used to retune
the checkpoint, gates or controller. Replays bind JSON and sensor-file digests.
These are integrity/lineage records, not trusted signatures or certification.
