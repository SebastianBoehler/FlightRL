# Unified robotics dashboard QA — 2026-09-06

final result: passed

## Scope and visual evidence

The accepted viewport-height robotics workbench is the visual target. This
change extends that same shell to all supported source types and removes the
separate page interfaces. It intentionally replaces “Other viewers” with one
environment/run selector. It does not redesign the simulation assets.

Source: `artifacts/workbench-layout-20260906/desktop-final.png` (1280x800).
Implementation: `artifacts/unified-dashboard-20260906/production-final.png`
(1002x1035). Both were opened in the same comparison input, showing production,
arm focus, dark theme and recorded inspection. Different viewport proportions
are explicitly accounted for; no pixel-fidelity score is claimed. The final
capture was repeated after capture zero finished decoding.

Additional opened evidence: `fleet.png` and `fleet-rerender.png` in the same
artifact folder. The latter shows all three actual re-rendered camera views.
Forest export was exercised and its 1536x1152 metadata checked. Map telemetry
was captured in `map-telemetry.png`. Viewport override screenshots showed stale
scaled browser-compositor layers and are excluded from visual acceptance.
Narrow-window verification below is DOM/interaction evidence, not a claim that
those corrupted images passed visual review.

## Findings and fixes

- P1: disconnected pages exposed inconsistent selectors, timelines and scrolling.
  Fixed: all six HTML entry points load the same application and shell. Sixteen
  catalog selections adapt the same camera, robot, history and timeline panels.
- P1 found during verification: forest fleet creation ran before scanned
  textures finished loading. Fixed by awaiting the materials before building
  that source. Target takeover, forest search, direct flight and original pilot
  subsequently reached their recorded endpoints without errors.
- P2: reconstruction feed inspection could revert to the initial camera after
  changing the selected camera. Fixed: its action now inspects the current
  camera, and labels/export paths follow the selection.
- P2: reconstruction overview reset geometry draw ranges. Fixed by immediately
  reapplying the selected timestamp after restoring the overview.
- P2: source-loading failures left generic connection text. Fixed with explicit
  source failure status while keeping the source selector usable.

## Required design surfaces

- Typography: existing system font, weights, numeric styling and hierarchy
  retained. Long source names truncate in the sidebar; full labels remain in
  the selector and workspace heading.
- Spacing: same fixed header/timeline, internally scrolling inspector and
  sensor panel. Feed columns follow stream count. Robot controls retain their
  sticky inspector header and use the existing form spacing.
- Colors: existing dark-green surfaces, muted borders, pale-green actions and
  selection state retained across source types.
- Images: source RGB is displayed with contain sizing and correct aspect ratio.
  Recorded actor images and newly rendered cameras are labeled separately;
  neither is substituted with generated or mock data. Renderer assets retained.
- Copy: source type, current robot, local/metric/arbitrary coordinate frame,
  unavailable tracking, playback provenance and simulation-only controls are
  visible. Unsupported saved replay in live forest is explicitly indicated.

## Functional checks

- TypeScript and production build pass. Only existing large Three.js bundle
  warnings remain; adapters are dynamically imported.
- All 16 catalog selections loaded. Every recorded source was exercised at an
  endpoint; empty-map/first-frame behavior was checked for learned mapping.
- Scene/Sensors/Telemetry, selected robot, signal selection, native arm commands,
  pause/save and stored-camera seek exercised. A 0.2 rad arm command produced
  a measured 0.197 rad transient in the shared history plot.
- Production and live forest showed approximately 30 display FPS and 10 RGB-D
  batches/s in the browser. This is an observed run result, not a GPU utilization
  measurement or broad hardware benchmark.
- Page size equaled viewport size at 1234x1035, 1422x889, 1280x800, 390x844 and
  1002x1035 CSS pixels. At 390x844, the sensor panel had 653 px client height and
  791 px content height; overflow remained within the component.
- Historical production URL normalized to the canonical root source URL.
  Build output confirms all six historical pages share the dashboard script.
- The robotics bridge finalizes on disconnect; renderer/observer/socket teardown
  occurs on page exit. Switching sources starts a new document/session.
- Final browser error log empty. Viewport overrides reset; production replay
  left open locally. No deployment, model-training claim or hardware test.

No outstanding P0/P1/P2 issue was found in the exercised source workflows.
A persistent cross-session run library and comprehensive assistive-technology /
multi-browser certification remain outside this change.
