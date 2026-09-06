import { panel, el } from "../dashboard/panel";
import * as T from "three/webgpu";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { disposeScene } from "../dispose-scene";
import { createForest } from "./scene";
import { Cameras, SIZES } from "./cameras";
import type { CameraRequest, WorldState } from "./types";

export async function start() {
  const ui = panel(), ids = ['drone-1', 'drone-2', 'drone-3'];
  ui.setup(ids.map(id => ({id, label: id, signals: [['height', 'Body height (m)'], ['speed', 'Body speed (m/s)']]})), ids.map((id, i) => ({id, label: `${id} · live RGB-D`, width: 256, height: 192, canvasId: `camera-${i + 1}`})));
  el('source-controls').hidden = false;
  el('source-controls').innerHTML = '<h3>Forest simulation</h3><button id="hover" disabled>Hold position</button><button id="dust" disabled>Dust demonstration</button><button id="drop" disabled>Drop debris</button><label><input id="rain" type="checkbox" disabled>Rain</label>';
  el('handover').textContent = 'Native flight dynamics and Jolt contacts. Hold and dust use a scene controller. Agras T25 reference · 32 kg unladen · 2.585 × 2.675 × 0.780 m. Authored geometry and estimated dynamics; no agricultural policy trained.';
  el("pose-title").textContent = "Body pose";
  const metric = el('metrics'), status = el('status');
  const fail = (error: unknown) => { el('error').textContent = String(error); };
  T.Object3D.DEFAULT_UP.set(0, 0, 1);
  const renderer = new T.WebGPURenderer({
    antialias: true,
    trackTimestamp: true,
  });
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = T.PCFSoftShadowMap;
  renderer.toneMapping = T.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.setPixelRatio(1);
  await renderer.init();
  if (!(renderer.backend as { isWebGPUBackend?: boolean }).isWebGPUBackend)
    throw Error("Live scene requires WebGPU");
  const view = document.getElementById("view")!;
  view.prepend(renderer.domElement);
  const camera = new T.PerspectiveCamera(55, 1, 0.03, 300);
  camera.up.set(0, 0, 1);
  camera.layers.enable(1);
  camera.position.set(-2, -7.2, 5.1);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(-2, 0, 1.2);
  controls.enableDamping = true;
  const resize = () => {
    if (!view.clientWidth || !view.clientHeight) return;
    const scale = Math.min(1, 1536 / view.clientWidth, 864 / view.clientHeight);
    renderer.setSize(Math.floor(view.clientWidth * scale), Math.floor(view.clientHeight * scale), false);
    camera.aspect = renderer.domElement.width / renderer.domElement.height;
    camera.updateProjectionMatrix();
  };
  const observer = new ResizeObserver(resize); observer.observe(view);
  resize();
  const world = await createForest(renderer),
    cameras = new Cameras(3, SIZES, ids.map(() => world.drone.camera_offset_m));
  let shown: WorldState | null = null;
  let state: WorldState = world.initial,
    pending: CameraRequest | null = null,
    drawing = false,
    lastDraw = 0;
  world.apply(state);
  await cameras.capture(renderer, world.scene, world.bodies, state);
  renderer.render(world.scene, camera);
  await renderer.waitForGPU();
  const socket = new WebSocket(`ws://${location.host}/__realism`);
  socket.binaryType = "arraybuffer";
  const send = (data: unknown) => {
    if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(data));
  };
  socket.onopen = () => send({ type: "scene", scene: world.description });
  socket.onerror = () =>
    fail(
      "Native connection failed. Start scripts/run_realism.py beside Vite on port 4173.",
    );
  socket.onclose = () => {
    status.textContent =
      "Native simulation disconnected. The last completed view is retained.";
  };
  let delivered = 0,
    lastCount = 0,
    reportAt = performance.now(),
    frames = 0,
    displayFps = 0,
    cameraHz = 0,
    physicsMs = 0, dustCount = 0;
  const frameTimes: number[] = [],
    gpuTimes: number[] = [];
  socket.onmessage = (event) => {
    try {
      const message = JSON.parse(event.data);
      if (message.type === "ready") {
        state = message.state;
        status.textContent = `Shared scene ${message.identity.scene_sha256.slice(0, 12)} · solid triangles ${world.description.triangleCount.toLocaleString()} · actor input is live RGB-D. Hold position and Dust demonstration use a scene controller. Agras T25 reference · 32 kg unladen · 2.585 × 2.675 × 0.780 m. Authored geometry and estimated dynamics; no agricultural policy trained.`;
        document
          .querySelectorAll<
            HTMLButtonElement | HTMLInputElement
          >("#source-controls button,#source-controls input,#pause")
          .forEach((b) => (b.disabled = false));
      } else if (message.type === "state") {
        state = message.state;
        if(state.notice)status.textContent = state.notice;
      }
      else if (message.type === "capture") pending = message;
      else if (message.type === "metrics") {
        delivered = message.camera_batches;
        dustCount = message.dust_airborne;
        physicsMs = message.physics_p95_ms;
        status.textContent = message.tracked.map((v: string, i: number) => `Drone ${i + 1}: ${v} · ${message.points[i].toLocaleString()} points`).join(' · ');
      } else if (message.type === "saved")
        status.textContent = `Captured observations and report saved to ${message.path}`;
      else if (message.type === "error") throw Error(message.message);
    } catch (error) {
      fail(error);
      socket.close();
    }
  };
  for (const [id, mode] of [
    ["hover", "hover"],
    ["dust", "dust"],
    ["pause", "paused"],
  ])
    document.getElementById(id)!.onclick = () => {
      if(mode === "dust") {
        world.guides.visible = true;
        camera.position.set(-2, -6.8, 2.3);
        controls.target.set(-4, 0, .8);
      }
      state.mode = mode;
      send({ type: "mode", mode });
    };
  el('overview').onclick = () => { world.guides.visible = true; camera.position.set(-2, -7.2, 5.1); controls.target.set(-2, 0, 1.2); };
  el('focus-robot').onclick = () => {
    world.guides.visible = false;
    const p = (shown ?? state).positions[ids.indexOf(ui.selection())];
    controls.target.fromArray(p); camera.position.copy(controls.target).add(new T.Vector3(3.4, -.6, 1.4));
  };
  const reference = document.createElement('p');
  reference.textContent = `${world.drone.name} · ${world.drone.payload}. Camera mount ${world.drone.camera_offset_m.join(' / ')} m; research RGB-D optics.`;
  el('source-controls').append(reference);
  el('replay-time').textContent = 'Live capture; saved replay not exposed by this bridge';
  document.getElementById("drop")!.onclick = () => send({ type: "drop" });
  document.getElementById("rain")!.onchange = (e) =>
    send({ type: "rain", enabled: (e.target as HTMLInputElement).checked });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      state.mode = "paused";
      send({ type: "mode", mode: "paused" });
    }
  });
  renderer.setAnimationLoop(async (now: number) => {
    if (drawing || document.hidden || now - lastDraw < 1000 / 30) return;
    drawing = true;
    lastDraw = now - ((now - lastDraw) % (1000 / 30));
    const began = performance.now();
    try {
      if (pending) {
        const request = pending;
        pending = null;
        world.apply(request.state);
        const result = await cameras.capture(
          renderer,
          world.scene,
          world.bodies,
          request.state,
        );
        for (let i = 0; i < 3; i++) {
          const canvas = document.getElementById(
            `camera-${i + 1}`,
          ) as HTMLCanvasElement;
          canvas
            .getContext("2d")!
            .putImageData(
              new ImageData(
                new Uint8ClampedArray(result.data[i * 4]),
                256,
                192,
              ),
              0,
              0,
            );
        }
        shown = request.state;
        ui.state({time_s: shown.time_s, robots: Object.fromEntries(ids.map((id, i) => {
          const q = shown!.quaternions[i];
          return [id, {position: shown!.positions[i], yaw: Math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)) * 180 / Math.PI,
            signals: {height: shown!.positions[i][2], speed: Math.hypot(...shown!.velocities[i])}}];
        }))});
        ui.captureLabel(`LIVE · ${shown.time_s.toFixed(3)} s · capture ${shown.sequence} · shared RGB-D acquisition`);
        send({ type: "camera", id: request.id });
        socket.send(result.packed);
      }
      if (shown) ui.captureLabel(`${state.mode === 'paused' ? 'PAUSED' : 'LIVE'} · ${shown.time_s.toFixed(3)} s · capture ${shown.sequence} · shared RGB-D acquisition`);
      world.apply(shown ?? state);
      controls.update();
      if (view.clientWidth) renderer.render(world.scene, camera);
      if ((renderer.backend as { trackTimestamp?: boolean }).trackTimestamp) {
        const duration = await renderer.resolveTimestampsAsync();
        if (duration !== undefined) gpuTimes.push(duration);
      }
      await renderer.waitForGPU();
      frameTimes.push(performance.now() - began);
      frames++;
      if (now - reportAt >= 2000) {
        displayFps = (frames * 1000) / (now - reportAt);
        cameraHz = ((delivered - lastCount) * 1000) / (now - reportAt);
        const sorted = [...frameTimes].sort((a, b) => a - b),
          p95 = sorted[Math.floor(sorted.length * 0.95)];
        metric.textContent = `${state.mode} · ${state.time_s.toFixed(1)} s · ${displayFps.toFixed(1)} display fps · ${cameraHz.toFixed(1)} RGB-D batches/s · physics p95 ${physicsMs.toFixed(1)} ms · ${state.contacts} contacts · ${dustCount} airborne dust parcels`;
        send({
          type: "display",
          fps: displayFps,
          camera_hz: cameraHz,
          p95_ms: p95,
          gpu_mean_ms: gpuTimes.length
            ? gpuTimes.reduce((a, b) => a + b) / gpuTimes.length
            : null,
          time_s: state.time_s,
          width: renderer.domElement.width,
          height: renderer.domElement.height,
          mode: state.mode,
        });
        frames = 0;
        lastCount = delivered;
        reportAt = now;
        frameTimes.length = 0;
        gpuTimes.length = 0;
      }
    } catch (error) {
      fail(error);
      renderer.setAnimationLoop(null);
      socket.close();
    } finally {
      drawing = false;
    }
  });
  window.addEventListener("pagehide", () => {
    observer.disconnect();
    socket.close();
    renderer.setAnimationLoop(null);
    cameras.dispose();
    controls.dispose();
    disposeScene(world.scene);
    world.disposeEnvironment();
    renderer.dispose();
  });
}
