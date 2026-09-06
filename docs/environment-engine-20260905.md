# Coupled environment engine

FlightRL owns the simulation and rendering path. This change does not depend on
Unreal, introduce an asset pipeline, or grant physical-flight authority.

`flightrl.environment.EnvironmentProfile` describes wind, turbulence, particle
transport, extinction, lights, windows and three surface materials. It is frozen,
validated and included in the inspection scene's content identity. The utility
plant has three artificial lights and no windows; window emission and sunlight
require explicit scene configuration. Native ray rendering reads the scene's
material and light buffers. Lens bloom, glare and grain remain a Metal camera pass.

The shared airflow field has seeded correlated gusts, spatial variation, four
rotor wakes aligned with the actual attitude, and a floor-impingement approximation.
The wake uses the previous native 20 ms step's thrust state. The drone samples
ambient air without its own wake, preventing double-counting of thrust. The native
six-DOF drag term is paired with wind acceleration to approximate drag relative to
the air. We retain the existing native flight integrator and renderer; vectorized
NumPy implements the reduced-order field and parcel transport.

Dust parcels respond to airflow with a relaxation time and prescribed terminal
settling velocity. Swept intersections stop them at equipment and partitions;
room-boundary contacts deposit them. Sufficient upward ground flow resuspends
settled floor parcels. Parcels are neither teleported through walls nor recycled
at room boundaries. The finite tracer count is conserved. Each parcel represents
a concentration weight, not a separately calibrated physical grain.

A 0.5 m concentration grid feeds twelve-sample camera-ray quadrature. Beer-Lambert
extinction and front-to-back scattering use the local concentration and cached
shadowed illumination from the same scene lights/windows. Camera RGB is processed
before policy downsampling. Depth remains ideal and is not labeled as a dust-aware
physical depth sensor. Both operator and onboard views use the recorded images;
the observer receives recorded active parcel positions.

This is a reduced-order model. It does not solve incompressible pressure, model
particle-particle collisions, motor contamination, abrasive damage or measured
particle size distributions. Wind disturbs flight; changing dust extinction alone
does not add an arbitrary force to the drone.

## Reproduce

```sh
PYTHONPATH=src .venv/bin/python setup.py build_ext --inplace --force
PYTHONPATH=src .venv/bin/python -m pytest tests/test_environment.py tests/test_inspection_industrial.py tests/test_inspection_scene.py tests/test_inspection_autonomy.py tests/test_inspection_replay.py tests/test_native_extension_sources.py -q
PYTHONPATH=src .venv/bin/python scripts/evaluate_environment_engine.py --checkpoint artifacts/utility-plant-optics-training-20260905/selected.pt --output artifacts/environment-engine-new
npm run build --prefix viewer
```

The evaluation freezes one existing policy, seed 400 and three conditions before
running. It does not retrain or silently select successful replays. Each condition
exports RGB-D, pose/airflow/particle telemetry, and training-ready RGB/depth,
proprioception and same-information teacher actions. Those datasets can be used by
the existing distillation workflow; a new broad training campaign is not claimed.

## Recorded regression result

All 43 focused tests passed; the viewer production build passed with the existing
large-bundle warning. Seed 400, frozen previously trained policy, 180 s budget:

| Condition | Panels | Collision | Mean transmission | Resuspensions |
|---|---:|---|---:|---:|
| Normal dust | 3/3 | No | 96.9% | 49 |
| Heavy dust | 1/3 | No | 77.9% | 40 |
| Stronger wind | 1/3 | No | 99.8% | 1 |

Initial transmission was 91.5% versus 39.2% for normal versus heavy dust at the
same initial pose. Mean transmission changes over the run as the camera moves and
particles deposit; it is not a matched-pose comparison. Episode wall times were
107, 99 and 111 s respectively, excluding packaging; these are local measurements,
not a general throughput benchmark. Three cases on one layout do not establish
robustness. The stronger-wind case exposes incomplete inspection despite ideal
depth and high transmission.

The full records, profile/source identities and training NPZs are under
`artifacts/environment-engine-20260905`. The visual review and first-20-second
camera clips are under `artifacts/environment-engine-review-20260905`.
