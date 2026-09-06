# Utility plant demonstration

The software demo now explores three connected equipment rooms, through offset service doors, with machinery and overhead pipe bounds. The classical planner only receives the camera, ideal depth, modeled odometry and link state. It does not receive the authored panel locations or room map.

The native camera renders 256 × 192 RGB with world-space materials and direct lighting. Aerosol extinction is exp(-0.035 × range), with 180 seeded, advected particles projected into those images. The policy receives the exact 4 × 4 average of the recorded RGB and sampled ideal range at 64 × 48. Observer equipment trim is presentation detail; navigation, range and collision use conservative boxes.

Correlated acceleration gusts act on velocity at each native 20 ms dynamics step: correlation time 0.6 s, stationary standard deviation [0.10, 0.10, 0.025] m/s². This is a simplified disturbance model, not CFD. Range remains ideal and odometry integrates modeled measured velocity with ideal attitude; this is not a calibrated optical sensor or visual localization pipeline.

The local training run uses eight training layouts, two validation layouts, three training seeds and corrective demonstrations. The selected checkpoint is frozen before four held-out utility-plant seeds are evaluated. These layouts vary equipment placement slightly, as well as independently seeded gusts and dust; they are not evidence of generalization to arbitrary buildings. The baseline and learned controller share the explicit altitude and local-clearance supervisor.

Replay exposes the actual recorded route, pose, image plane, camera feeds, gust acceleration and operator link. Playback wraps to the beginning while preserving the chosen view. The light/dark toggle affects both the interface and observer background. `review/review.html` is the older static review; open `/` for the live workbench.

Reproduce from the FlightRL root:

```sh
PYTHONPATH=src .venv/bin/python setup.py build_ext --inplace --force
PYTHONPATH=src .venv/bin/python scripts/train_inspection_student.py --industrial --output artifacts/plant-training-new
PYTHONPATH=src .venv/bin/python scripts/evaluate_utility_plant.py --checkpoint artifacts/plant-training-new/selected.pt --output artifacts/plant-evaluation-new
npm run build --prefix viewer
```

Local results: `artifacts/utility-plant-training-20260905/training.json` and `artifacts/utility-plant-evaluation-20260905/evaluation.json`. Images and sensor NPZ recordings are exported alongside each episode. The original room results remain a separate earlier benchmark.

## Enhanced camera pass

A second image pipeline adds world-space equipment instruments, ceiling windows, directional sunlight with obstacle shadows, and roughness-dependent glossy highlights. A Metal compute shader applies bright-source bloom, restrained horizontal glare, vignetting and deterministic sensor grain to the actual dust-attenuated recording before policy downsampling. The overview receives none of these image effects. This is an authored procedural renderer, not photorealistic path tracing or calibrated sensor simulation.

The expanded optical appearance gets its own retraining run, `artifacts/utility-plant-optics-training-20260905`, and fresh evaluation seeds 300–303. Previous utility-plant results (including the initial doorway collision and a missed learned inspection) remain in the earlier artifact directories. Scan position hold and a 0.10 m waypoint tolerance address the diagnosed drift into a doorway edge; increasing map inflation was rejected because it prevented exploration through otherwise navigable passages.

Final enhanced-camera check: seeds 300–303, 180 s budget each. Learned controller:
12/12 panels across four runs, zero collisions. Classical: 11/12 panels, zero
collisions. The single tested learned link-loss rollout reconnected without a
collision. Full geometric coverage can coexist with `budget_exhausted`: the
observation-only planner is not told the hidden total panel count. These are
narrow authored-scene results, not broad navigation reliability estimates.

Verification: 29 focused tests pass, viewer production build passes (existing
large WebGPU bundle warning). Actual browser checks cover both themes, looping,
recorded camera-plane alignment and cutaway pose inspection. Visual review and
compressed onboard footage: `artifacts/utility-plant-review-20260905/review.html`.
