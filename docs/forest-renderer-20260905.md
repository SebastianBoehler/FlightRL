# Detailed forest renderer — 2026-09-05

Open http://127.0.0.1:4173/forest.html with the viewer Vite dev server.
The environment review links to this page and displays actual camera exports.

## Implemented

- Tapered branching trees with roots and merged woody geometry.
- 173,964 instanced alpha-cutout leaves in the current seeded scene.
- Procedural bark, soil and litter textures, instanced grass and rocks.
- Directional canopy shadows, hemisphere lighting and distance fog.
- Shader leaf sway and 160 seeded falling leaves with wind drift and ground settling.
- Recorded drone camera poses, looping playback and orbit exploration.
- 1536 × 1152 PNG exports with camera pose, FOV, source hash and training provenance.

The export button writes locally through Vite development middleware to
artifacts/forest-quality-20260905. A static production host does not provide
this endpoint and reports an error if saving is attempted.

## Limits and validation

This is a procedural visual renderer, not a photorealism or physics-validation claim.
Leaf motion is a reduced-order animation, not resolved aerodynamics.
Additional foliage and ground detail have not been integrated into native collision
or the training observation renderer. Prior generalization results used the original
native RGB-D camera. These new exports are RGB only and have not trained a policy.

TypeScript and production build pass. Browser exports were checked at 1536 × 1152
from departure, observer and a later recorded camera pose.
The displayed frame counter measures animation-loop cadence, not GPU completion,
physics throughput or training steps per second.

Next: make detailed geometry available to the sensor renderer and collision
representation, export aligned depth, benchmark synchronized camera steps and rerun
held-out policy evaluation. Tree variation, understory and lighting need further
fidelity work before calling the forest realistic.

## Replay camera delivery fix — 2026-09-06

The mission viewer now bounds the observer to 30 frames per second and waits for
GPU completion before submitting another observer frame. The camera renderer also
waits for GPU completion before copying into the onboard and operator canvases.
Three's `renderAsync` alone only initializes and submits rendering; it does not
wait for completion. The latest-pose queue retains the final requested pose during
scrubbing and publishes completed frames during continuous replay.

The mission panel distinguishes the recorded operator link from local camera
freshness. The camera indicator reports delivered frames per two-second window,
CPU submission plus GPU completion elapsed time, and a delay warning when a pending
frame has not completed for 500 ms. These are presentation metrics, not simulation
or training throughput. Paused playback naturally reports zero new frames.

Validation: TypeScript/Vite build and both latest-pose queue regressions pass.
In the actual in-app browser, continuous forest replay delivered about 18 fps at
2x and 32 fps at 4x with sampled displayed timestamps within 0.1 replay seconds.
This is a local observation, not a guarantee against stalls on other hardware.

### Camera canvas presentation correction

A subsequent user report showed that submission/completion counters alone were
insufficient evidence of image freshness. The onboard panel now displays the
WebGPU renderer canvas itself. Operator and in-scene camera images are copied
synchronously immediately after rendering, before yielding to GPU completion.
This avoids reading the browser-managed canvas presentation buffer after an
asynchronous wait. GPU completion still bounds subsequent submissions. Switching
back to a recorded episode restores its original 2D canvas.

### Shared renderer supersedes the separate feed loop

The separate camera renderer and async queue were subsequently removed from the
single-drone presentation path. `ForestFeed.flush` now renders the newest requested
pose, copies its image and restores viewport/scissor state inside the overview's
render callback, using the same WebGPU renderer. The overview then renders before
the single GPU-completion wait. WebGPU viewport coordinates use the top-left origin.
This also avoids independent renderer state for the shared forest materials.
Earlier delivery-counter checks did not establish the cause of intermittent freezes.
