# Shared forest renderer and contacts — 2026-09-06

The live forest now feeds its displayed geometry into native contacts and renders
actual RGB-D observations for the existing sensor actor. Open
`http://127.0.0.1:4173/realism.html`. This is a local simulation; the frozen actor
has not been retrained for forest imagery and is not a successful autonomy demo.

## Implemented

- Three.js WebGPU rendering with scanned bark/soil color, normal and roughness/AO
  maps, HDR environment lighting, foliage transmission and a 4096² sun shadow.
  Three photogrammetry stumps add detailed solid geometry. The 12 local asset
  files total about 11 MiB; `viewer/public/assets/forest/manifest.json` records
  source URLs, CC0 licenses, byte sizes and SHA-256 hashes. No runtime CDN needed.
- The current scene exports 370,274 solid triangles in metres, Z-up, plus explicit masses,
  body dimensions and xyzw poses. Trunks, branches, terrain, cabin, rocks, stumps
  and beacons enter Jolt. Soft canopy/grass/ferns render in RGB-D without hard
  collision. Drones use their box envelopes; debris uses matching box geometry.
- Jolt 5.3.0 is pinned to commit `0373ec0dd762e4bc2f6acdb08371ee84fa23c6db`.
  A small C++/ctypes bridge provides CCD, friction, restitution, contacts and ray
  queries. Physics stays at 50 Hz. Existing native thrust, rate response and drag
  feed the solver; gravity is integrated once. Inertia comes from each box's
  mass and dimensions, rather than identified hardware inertia.
- Three cameras capture at 10 Hz. Each independently renders 64 × 48 RGB-D for
  the frozen actor and 256 × 192 for RGB-D visual odometry and mapping. All six
  views use the same timestamped scene. Depth is ray distance in metres, capped
  at 8 m; images have a top-left origin, 63° vertical FOV and body offset
  `[0.035, 0, 0.012] m`. Rendering completes before readback and delivery.
- Actor inputs retain RGB-D, body proprioception, role and delayed visual reports.
  Actor inference and three empty-start reconstructions run in both active modes.
  Only Experimental policy applies inferred controls. Hold position and Dust
  demonstration use an explicit position/attitude controller with simulator state;
  they do not claim learned autonomous behavior.
- A finite bed of 1,024 dust parcels and 160 leaves uses gravity, drag, shared
  reduced-order wind/downwash and swept Jolt ray contact. Contact events can lift
  settled dust. Settled particles follow moving supports. Rain has 320 reusable
  emitter slots, with emitted/impact counters. Dust accounting conserves parcel
  count; rendered parcel size does not assert literal grain size or measured mass.
- The presentation is capped at 30 Hz and 1536 × 864 internal pixels while
  preserving viewport aspect ratio. Canopy/particles are instanced, textures use
  mipmaps, shadows update once per sensor batch, and only one camera batch can be
  pending. Hidden tabs pause physics; timing failures surface as errors.
- Pause & save writes scene/actor/source fingerprints, timing and particle
  counters, and compressed observations. The first 50 batches per mode include
  both image resolutions, proprioception, roles, messages, inferred and applied
  actions, capture sequence/time and application time. This is bounded capture,
  not a complete episode recorder or a training run.

## Run locally

From the FlightRL checkout, with CMake and a C++ compiler installed:

```sh
uv pip install --python .venv/bin/python -e '.[realism]'
.venv/bin/python scripts/build_realism_physics.py
PYTHONPATH=src .venv/bin/python scripts/run_realism.py \
  --actor artifacts/camera-control-linkloss-20260906/actor.pt \
  --output artifacts/realism-local-run
```

In a second terminal, `npm run dev --prefix viewer -- --port 4173`.
The output directory must be new. The existing forest episode at
`viewer/public/data/forest-held-out.json` and the local actor checkpoint are
required and remain ignored research artifacts. The bridge listens only on
127.0.0.1:8766 and accepts the local viewer origin. This checkout-based native
build is separate from the original simulator extension.

## Validation

Evidence: `artifacts/realism-implementation-20260906/`. The `before/` snapshot
preserves the starting renderer. Runtime sources and asset hashes identify this
uncommitted implementation; this is not a clean-commit research promotion.

- 23 focused tests passed: native contacts plus existing camera actor boundary,
  fleet contract, reconstruction and dust contact regressions. Contact checks
  cover gravity/hover, repeatability, resting friction, 200 m/s thin-wall CCD,
  metric rays, invalid input, moving particle supports and finite dust/rain counts.
- Browser GPU calibration passed 18 camera/resolution cases: translated poses,
  sloped surfaces and rotated cameras. Maximum ray-distance error was 0.033 mm;
  empty views returned exactly 8 m. The archived calibration HTML/TypeScript can
  be copied back to `viewer/` and `viewer/src/realism/` to rerun the probe.
- TypeScript check and Vite production build passed. Vite reports its existing
  large Three.js chunk warning.

### Combined-load result on the M4 Max

The final 103.78-second run used 1536 × 863 internal pixels (the 864-pixel cap
rounds down by one pixel at this viewport), three drones, three debris bodies,
372,330 solid triangles, canopy animation, rain and dust. Policy inference and
all three reconstructions remained enabled. There were 1,038 completed camera
batches and 5,189 fixed physics steps.

| Measurement | Result |
| --- | --- |
| Display FPS, median / lowest 2-second sample after warmup | 30.00 / 29.87 |
| Completed three-camera batches/s, median / lowest sample | 10.00 / 9.84 |
| Physics including particles, median / p95 | 2.69 / 3.20 ms |
| Camera request through readback, actor and mapping, median / p95 | 41.29 / 52.69 ms |
| Median of display-cycle p95 wall times | 22.90 ms |
| Median reported mean GPU pass duration | 6.58 ms |

GPU duration covers timestamped render passes; it is not a machine utilization
percentage and does not include every readback/copy or native compute operation.
The roughly 10 ms difference between the display-cycle p95 and the 33.3 ms
presentation budget is a local timing margin, not a guaranteed capacity multiplier.
The backend resident footprint sampled during the run was about 559 MiB;
shared browser GPU memory was not separately attributed.

The run recorded 154 contact-added events and 24,580 rain impacts. All 512 dust
parcels were accounted for: 11 airborne, 501 settled, zero escaped at save.
Resuspension counted repeated lift events, not newly created dust. Replaying all
100 saved actor packets (50 hover, 50 policy) reproduced inferred actions exactly.
Applied actions matched policy output in policy mode and zero CTBR offsets in
hover mode; capture/application timestamps and finite metric depths were checked.
The 390 × 844 layout had no horizontal overflow and its controls were exercised.

The stable benchmark summary is `verified-summary.json`; full session data is
`verified/bbe34eff/` under the evidence directory. The UI was paused after the
run; later control checks do not extend this benchmark. Browser error logs were
empty. Final policy views lost tracking, so no mission-success claim is made.

## Limits

This establishes the shared rendering/physics path and a first visual upgrade;
full Unreal feature parity is not implemented. There is no Lumen/Nanite,
ray-traced global illumination, full fluid solve, wet-surface model, deformable
vegetation or identified leaf aerodynamics. Downwash uses an approximate floor
at -0.04 m; exact geometry governs particle contacts, not the airflow pressure
field. Sensor exposure is fixed and sensor noise is not calibrated.

The frozen policy can collide and lose visual tracking in this forest. The new
observations are usable by the actor network, but training, held-out mission
success and sim-to-real validation remain separate work. Existing historical
recordings and native training environments are not silently converted to this
scene. Long-duration memory growth and 60 FPS are not established by this test.

## Forest layout and flight visibility follow-up

The old bounded trunk plot plus separate background ring was replaced by one
seeded stand of 108 trees. Heights vary from 3.4–11.2 m, with correlated trunk
thickness, lean and crown spread, irregular spacing and soft clearing density.
The cabin and initial drone positions have clearance. Exported live collisions
use the new trunks and branches. The earlier 103.78-second benchmark above remains
identified as the prior 372,330-triangle scene.

All three drones now have color-matched labels, body heading arrows and rolling
position trails. Histories store at most 600 samples per drone at up to 10 Hz;
annotations remain visible through foreground geometry and use observer-only
render layers. Sensor images exclude these aids. Recorded-flight renderers share
the revised vegetation layout, but historical trajectories retain their original
physics; live RGB-D and Jolt are the authoritative shared scene.

The revised scene ran for 69.44 seconds at the normal 1194 × 600 viewport:
median 30 FPS, 10 completed camera batches/s, physics p95 3.28 ms and full
camera/actor/mapping latency p95 68.10 ms. This is a separate measurement from
the earlier near-1536 × 864 benchmark. `natural-forest-summary.json` binds the
result to scene `0ffad1b039c2` and session `verified/af9ee9cf/`. The GPU isolation
check compared all 12 sensor attachments with guides present versus absent:
identical bytes. After 800 updates, each of the three trails still held exactly
600 samples. See `guide-isolation.json` and its archived browser harness.

## Controlled dust and continuous terrain follow-up

The visible HDR photograph was replaced with Three.js atmospheric sky rendering;
its sun direction follows the scene's directional light. HDR image-based lighting
is retained. Radial terrain keeps dense near-field geometry, then becomes coarse
rolling hills out to 1.2 km, with atmospheric haze. World-space texture coordinates
keep the soil scale continuous. The entire mesh enters Jolt, and scene validation
allows coordinates up to 2,048 m. This removes the finite square floor's boundary
against unrelated photographed terrain while preserving free observer movement.

Hold position is now a real position/attitude feedback controller through the same
native actuator/contact integration. Dust demonstration descends smoothly over six
seconds to about 0.48 m, then makes a small horizontal sweep. It also brings the
observer camera closer; orbit controls remain available. The finite dry bed contains
1,024 parcels concentrated around the three hover locations. Instanced camera-facing
sprites with stochastic coverage give softer dust silhouettes at a fixed cost.
This is approximate optical coverage of parcels, not measured aerosol scattering
or a volumetric fluid solver. Settled particles remain finite and can be depleted
from a location by wind. Particle rendering participates in sensor images; dust
can consequently disrupt visual odometry. No mapping-success claim is implied.

The original frozen policy was trained on different images and can command a
persistent climb. It is labeled Experimental policy, and pauses beyond 6 m altitude,
below -0.5 m, or over 12 m from an initial position. The UI explains why it paused.
This bound does not train or repair the policy and adds no invisible collision wall.
The new controller is deliberately distinct from policy inference. Earlier zero-CTBR
hover benchmark results above describe the previous version only.

A 40-second native regression verifies wind rejection, controlled descent, bounded
low flight and dust lifting. Seven other contact regressions remain passing. An
actual Session/Jolt boundary check paused at 6.2 m; evidence is in
`policy-envelope-check/result.json`. The viewer build and live sensor delivery
passed, with no browser error logs. The industrial environment was not part of
this dust-focused change.

The final dust run lasted 128.24 seconds at 1179 × 600: median 30 FPS and
10 camera batches/s, physics p95 4.10 ms, camera/actor/mapping latency p95
61.91 ms. All 1,024 dust parcels were settled at the end, after 1,619 lift events;
none escaped. Reset scene starts a fresh authored bed and flight state. All saved
RGB-D depths were finite and in range. The focused regression set passed 24 tests.
See `dust-summary.json`, `dust-tests.log`, and `dust-demo/de727304/`.
