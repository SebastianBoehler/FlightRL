# Inspection replay workbench

Local WebGPU presentation of actual FlightRL mission recordings. It is outside
simulation/training. No backend service, cloud deployment or live-drone authority.

```sh
npm ci --prefix viewer
mkdir -p viewer/public
ln -s ../../artifacts/environment-engine-20260905 viewer/public/data
npm run dev --prefix viewer -- --port 4173
```

Open http://127.0.0.1:4173. `viewer/public/data` is an ignored symlink to generated
artifacts. To produce a different recording, run `scripts/evaluate_environment_engine.py`
with a frozen checkpoint and new output directory, then point the symlink there.
WebGPU is required; errors are surfaced rather than switching rendering backends.

The scene uses recorded room/box/panel geometry and native world-from-body camera
transforms. Two walls are cut away only in the observer view. Display materials,
vent overlays, a selection halo and floor markings are presentation styling;
the camera inset is unmodified recorded RGB. The image plane is 0.85 m from the
camera and the displayed frustum is truncated at 1.4 m for readability; the
recorded RGB-D camera's range limit is 8 m. The two renderers share geometric
inputs and pose, not identical shading.

Use episode selection, timeline, chapter buttons and overview/follow/camera-pose
views. During link loss, operator video is unavailable. The separately labeled
onboard recording remains viewable retrospectively; this is not a live operator
feed through a failed link. Validated panel counts come from the evaluator, not
from information given to the mission planner.

The utility plant has three connected equipment rooms. The recorded RGB includes
native material lighting and glossy highlights, dust extinction/particles and a
Metal lens pass (bloom, mild glare, vignette and grain). These sensor effects are
applied before training downsampling; the WebGPU observer does not apply them.
The actor receives 64 × 48 RGB-D derived from 256 × 192 RGB and ideal range.
Generating the enhanced camera requires PyTorch MPS with `compile_shader` on the
local Mac; replay itself only requires WebGPU.

Playback loops. Chapter selection pauses at that event. Theme switching changes
the interface and observer background. Camera-pose mode fades scene equipment to
make the recorded image plane visible. The scene remains orbitable in overview.

The evaluation report includes every held-out seed and missed inspection. Failed
or incomplete recordings are available in episode selection. This is procedural
simulation with authored equipment variations, ideal depth and modeled odometry;
it is not photorealistic defect imagery or evidence of physical-flight transfer.
See `docs/utility-plant-demo-20260905.md` for reproduction and assumptions.

The coupled environment replay adds measured camera transmission, ambient wind,
airborne and settled parcel counts. Profiles, limitations and regression commands
are documented in `docs/environment-engine-20260905.md`.
