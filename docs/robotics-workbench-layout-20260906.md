# Robotics workbench: viewport application layout

2026-09-06. Replaces the vertically accumulated robotics page with a full-height
application shell on `robotics.html`. Existing forest, fleet and reconstruction
viewers remain separate routes, linked as Other viewers.

## Interaction model

- The app uses the window's dynamic height. Header, timeline and status stay
  outside the independently scrolling workspace regions. The page never scrolls.
- Scene keeps the 3D viewport central, with a compact camera strip. Sensors opens
  full camera feeds. Telemetry shows the selected robot's recorded signal history.
- Inspector owns robot selection, camera pose, joint setpoints, handover and link
  control. Its selection/header stay visible while the contents scroll. Inspector
  toggles from the toolbar; Escape closes it and returns focus to that button.
- Inspect on a feed selects its robot and opens the inspector. Focus selected robot
  returns to Scene and positions the existing observer camera near that robot.
- Every view shares the acquisition timeline. Historical actuator targets and
  readings remain tied to the selected recorded capture.
- Telemetry offers camera height/measured speed for drone/rover and joint-one
  position, velocity or actuator effort for the arm. These use recorded state.
- At narrow widths, Inspector becomes an overlay. At short heights, the scene's
  thumbnail strip is hidden; all feeds remain available in Sensors. Panels can
  scroll horizontally or vertically without expanding the document.

## Implementation boundary

`viewer/src/robotics/shell.ts` owns the shell and navigation; `workbench.ts` binds
robot data; `history.ts` draws signal history; the two workbench stylesheets own
layout and responsive rules. Robotics no longer inherits the forest page layout.
No new UI framework or dependency was added. Other viewer styles are unchanged.
The observer renderer is skipped when its panel is hidden; metric camera capture
continues at its existing cadence. Resizing a hidden panel cannot allocate a
zero-size renderer or produce an invalid camera aspect ratio.

## Verification

- Production viewer build and TypeScript checks pass. Existing Three.js chunk-size
  warnings remain. No new physics/learning test suite was needed for this UI change.
- Browser checks covered Scene/Sensors/Telemetry, robot selection, arm commands,
  signal selection, inspector toggle/Escape, pause/save and timeline replay.
- A 0.25 rad arm command settled at 0.250 rad while navigating sensor/telemetry
  panels; replay of capture zero restored its initial target and state.
- Final inspector-header check used a 0.2 rad command. At 390x844, the inspector
  scrolled 442.5 px, its sticky header remained at y=55 px, and page scroll was 0.
- Measured document dimensions exactly matched 1280x800, 1280x500 and 390x844.
  In the 500 px-high window the 3D region retained 310 px of height.
- Browser error log empty. Scene measurements observed roughly 30 FPS and 10
  camera batches/s; this is a UI smoke check, not a new thermal/performance suite.
- Evidence is under `artifacts/workbench-layout-20260906/`; visual review is in
  `design-qa.md`. Temporary viewport overrides are reset after verification.

## Next

Define one arm task with independent success and contact checks, then establish
its reference-controller baseline before training. Use an industrial partner's
representative task and model to test the product value. Consolidating the older
viewer routes into this shell is a separate UI migration, not completed here.


Follow-up: all source interfaces are now consolidated; see
[Unified dashboard](unified-dashboard-20260906.md). The separate-viewer scope
below describes the earlier layout milestone.
