/** One viewport, independently scrolling work areas, and one shared run clock. */
export function mountShell() {
  document.getElementById("app")!.innerHTML = `
    <header class="app-bar">
      <div class="brand">SUNDERLABS<span>Robotics studio</span></div>
      <div class="project-title"><h1>Robotics workbench</h1><p>Inspection & model validation</p></div>
      <span class="environment-label">Simulation</span>
      <div class="run-actions"><button id="reset">Restart</button><button id="pause" class="primary" disabled>Pause & save</button><button id="toggle-inspector" aria-expanded="true" aria-controls="inspector">Inspector</button></div>
    </header>
    <main class="workspace" data-view="scene">
      <aside class="sidebar" aria-label="Workspace navigation">
        <span class="section-label">Workspace</span>
        <nav aria-label="Workbench views">
          <button data-view="scene" aria-pressed="true">Scene</button>
          <button data-view="sensors" aria-pressed="false">Sensors</button>
          <button data-view="telemetry" aria-pressed="false">Telemetry</button>
        </nav>
        <div class="sidebar-context"><span class="section-label">Environment</span><select id="source-select" aria-label="Environment and run" disabled><option>Loading sources…</option></select><p>Switching starts a new session.</p></div>

        <span class="sidebar-note">Local simulation<br>Recorded evidence</span>
      </aside>
      <section class="stage" aria-label="Simulation workspace">
        <div class="stage-bar"><h2 id="view-title">Scene</h2><div class="view-actions"><button id="overview">Overview</button><button id="equipment" hidden>Equipment view</button><button id="focus-robot">Focus selected robot</button></div></div>
        <div id="view" aria-label="Interactive 3D scene"><div id="metrics">Connecting to simulation…</div><span class="viewport-hint">Drag to orbit · scroll to zoom</span></div>
        <section id="sensor-panel" class="scroll-panel" aria-label="Camera feeds"><div class="panel-heading"><h3>Camera feeds</h3><span>Source images · acquisition provenance</span></div><div id="feeds"></div></section>
        <section id="telemetry-panel" class="scroll-panel" aria-label="Sensor history" hidden><div class="panel-heading"><div><h3>Signal history</h3><p>Selected robot · source timestamps</p></div><label>Signal <select id="signal-select" aria-label="Telemetry signal"></select></label></div><canvas id="history" width="1000" height="400" role="img" aria-label="Selected robot signal history"></canvas><p id="history-label">Waiting for sensor captures</p></section>
      </section>
      <aside id="inspector" class="scroll-panel" aria-label="Robot inspector">
        <div class="inspector-top"><div class="panel-heading"><h2>Inspector</h2><button id="close-inspector" aria-label="Close inspector">Close</button></div>
        <label class="field-label" for="robot-select">Selected robot</label><select id="robot-select"></select></div>
        <section class="inspector-section"><h3 id="pose-title">Pose</h3><p id="pose-frame">World frame · metres / degrees</p><dl id="robot-pose"><div><dt>X</dt><dd>—</dd></div><div><dt>Y</dt><dd>—</dd></div><div><dt>Z</dt><dd>—</dd></div><div><dt>Yaw</dt><dd>—</dd></div></dl></section>
        <section id="source-controls" class="inspector-section" hidden></section><details id="arm-panel" class="inspector-section" hidden><summary>xArm7 · joints and actuator controls</summary><p>Joint servos in radians. Gripper uses the source model’s 0–255 scale.</p><div id="joints"></div><button id="apply-arm" class="primary">Apply arm setpoints</button></details>
        <section class="inspector-section"><h3 id="details-title">Run details</h3><div id="handover">Waiting for the first observation.</div></section>
        <section id="communication-panel" class="inspector-section" hidden><h3>Communication</h3><label class="check-label"><input id="link" type="checkbox" checked disabled> Robot link</label><p>Controls the simulated inspection handover.</p></section>
      </aside>
    </main>
    <section class="run-timeline" aria-label="Run timeline"><div class="timeline-heading"><strong>Run timeline</strong><span id="capture-time" role="status">Waiting for the first capture</span></div><div class="timeline"><label for="timeline">Source time</label><input id="timeline" type="range" min="0" max="0" value="0" disabled><select id="speed" aria-label="Playback speed" hidden><option value="1">1×</option><option value="2">2×</option><option value="4">4×</option></select><output id="replay-time">Pause & save to replay</output><button id="live" disabled>Latest capture</button></div></section>
    <footer class="status-bar"><span class="status-label">RUN</span><div id="status" tabindex="0">Connecting to the simulation bridge…</div></footer><div id="error" role="alert"></div>`;
  const workspace = document.querySelector<HTMLElement>(".workspace")!;
  const inspector = document.getElementById("inspector")!;
  const toggle = document.getElementById("toggle-inspector")!;
  const navigation =
    document.querySelectorAll<HTMLButtonElement>("nav [data-view]");
  function showInspector(open: boolean) {
    inspector.hidden = !open;
    workspace.classList.toggle("inspector-closed", !open);
    toggle.setAttribute("aria-expanded", String(open));
  }
  toggle.onclick = () => showInspector(inspector.hidden);
  document.getElementById("close-inspector")!.onclick = () => {
    showInspector(false);
    toggle.focus();
  };
  function showView(name: string) {
    workspace.dataset.view = name;
    document.getElementById("view-title")!.textContent =
      name[0].toUpperCase() + name.slice(1);
    document.getElementById("telemetry-panel")!.hidden = name !== "telemetry";
    navigation.forEach((button) =>
      button.setAttribute("aria-pressed", String(button.dataset.view === name)),
    );
  }
  navigation.forEach(
    (button) => (button.onclick = () => showView(button.dataset.view!)),
  );
  ["overview", "equipment", "focus-robot"].forEach((id) =>
    document
      .getElementById(id)!
      .addEventListener("click", () => showView("scene")),
  );
  const narrow = matchMedia("(max-width: 900px)");
  showInspector(!narrow.matches);
  narrow.addEventListener("change", () => showInspector(!narrow.matches));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !inspector.hidden) {
      showInspector(false);
      toggle.focus();
    }
  });
}
