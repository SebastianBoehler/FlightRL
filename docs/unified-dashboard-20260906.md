# Unified robotics dashboard — 2026-09-06

The viewer now has one application entry point (`viewer/src/dashboard.ts`) and
one viewport-height shell. Source selection replaces links to separate viewers.
The catalog includes both industrial simulations, live forest physics, four
single-drone recordings, the detailed forest render, six fleet recordings, and
two reconstruction sources (16 selections total).

## Boundaries

A source adapter owns its scene renderer and source protocol. Shared components
own navigation, robot selection, camera tiles, pose inspection, history plots,
and recorded playback. Robot-specific joint controls and reconstruction tools
appear only on sources that supply them. No new physics engine, generic model
import format, or hardware clock guarantee is implied by this UI consolidation.

The canonical URL is `/?source=<id>`. All six historical HTML entry points load
the same dashboard and normalize their existing query parameters to this URL.
Changing sources performs a document navigation: the old socket/renderers are
closed, and the next source starts a new session. The robotics bridge finalizes
its recording on disconnect. A persistent recording library and in-place session
switching are outside this change.

## Evidence and timing

- Recorded playback follows source sample times, stops at the end, and seeks
  exact samples. Scene, sensor pixels, pose and plot cursor use that selection.
- Robotics replay still decodes the entire stored camera batch before accepting
  a seek, ignoring stale responses. Its original MCAP acquisition timestamps and
  raw/delayed observation distinction are unchanged.
- Fleet sources with sensor atlases show original policy pixels. Other fleet
  sources explicitly show re-renders at recorded poses. Single-drone sensor
  panels now retain their original pixels even with a detailed forest scene.
- Reconstruction retains local frames, RGB arbitrary scale, evaluation-only
  reference trajectories, and tracking gaps. Plots do not bridge missing poses.
- Live forest aligns its displayed body state with the last completed RGB-D
  acquisition. It exposes live controls and saving; its bridge does not provide
  stored-image replay, which is explicitly indicated in the timeline.

Only the chosen adapter is imported. Observer rendering is skipped when hidden;
WebGPU animation is capped at 30 Hz. Existing Three.js bundle-size warnings remain.

Browser evidence and responsive captures are under
`artifacts/unified-dashboard-20260906/`; see root `design-qa.md` for verification.
