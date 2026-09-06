import { replayControls } from "./replay";
import { workbench } from "./workbench";
import * as T from "three/webgpu";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { Cameras } from "../realism/cameras";
import { industrialScene } from "./scene";
import { FlightGuides } from "../realism/flight-guides";
import { inspectionStatus } from "./inspection-status";
import { siteQuery } from "./site-selection";
import type { RobotState, RobotMessage } from "./types";
import { disposeScene } from "../dispose-scene";

export async function start() {
  const panel = workbench();
  document.getElementById("pose-title")!.textContent = "Camera pose";
  document.getElementById("communication-panel")!.hidden = false;
  document.getElementById("equipment")!.hidden = false;
  T.Object3D.DEFAULT_UP.set(0, 0, 1);
  const renderer = new T.WebGPURenderer({ antialias: true });
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = T.PCFSoftShadowMap;
  renderer.toneMapping = T.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.setPixelRatio(1);
  await renderer.init();
  if (!(renderer.backend as any).isWebGPUBackend)
    throw Error("WebGPU is required for metric sensor rendering");
  const view = document.getElementById("view")!;
  view.prepend(renderer.domElement);
  const camera = new T.PerspectiveCamera(55, 1, 0.02, 300);
  camera.up.set(0, 0, 1);
  camera.position.set(-6, -2.8, 5.5);
  camera.layers.enable(1);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0.5, 0, 1);
  controls.enableDamping = true;
  const resize = () => {
    if (!view.clientWidth || !view.clientHeight) return;
    const scale = Math.min(1, 1536 / view.clientWidth, 864 / view.clientHeight);
    renderer.setSize(
      Math.round(view.clientWidth * scale),
      Math.round(view.clientHeight * scale),
      false,
    );
    camera.aspect = view.clientWidth / view.clientHeight;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(view);
  resize();
  let cameras: Cameras;
  let sensorBodies: T.Object3D[] = [];
  let shown: RobotState | null = null;
  let replaying = false;
  let closeUp = false;
  let sensorNames: string[] = [];
  const socket = new WebSocket(`ws://${location.host}/__robotics${siteQuery()}`);
  let world: Awaited<ReturnType<typeof industrialScene>> | null = null,
    pending: Extract<RobotMessage, { type: "capture" }> | null = null,
    state: RobotState | null = null,
    guides: FlightGuides | null = null,
    drawing = false,
    loading = true,
    assets: Array<{ id: number; asset?: string }> = [],
    completed = false,
    last = 0,
    frames = 0,
    reported = performance.now(),
    delivered = 0,
    previous = 0;
  const send = (value: unknown) => {
    if (socket.readyState === WebSocket.OPEN)
      socket.send(JSON.stringify(value));
  };
  const replay = replayControls(send);
  const fail = (error: unknown) => {
    document.getElementById("error")!.textContent = String(error);
  };
  socket.onmessage = async (event) => {
    try {
      const message: RobotMessage = JSON.parse(event.data);
      if (message.type === "scene") {
        loading = true;
        completed = false;
        pending = null;
        (document.getElementById("pause") as HTMLButtonElement).disabled = true;
        (document.getElementById("link") as HTMLInputElement).disabled = true;
        while (drawing) await new Promise(requestAnimationFrame);
        if (world) {
          disposeScene(world.scene);
          world.disposeEnvironment();
        }
        cameras?.dispose();
        cameras = new Cameras(message.description.cameras.length, [[512, 384], [128, 96]]);
        panel.setup(message.description.cameras.map((c) => c.robot_id));
        world = await industrialScene(renderer, message.description);
        const reference = message.description.drone_reference;
        const info = document.getElementById("source-controls")!;
        info.hidden = false;
        info.textContent = `${reference.name} · ${reference.mass_kg} kg · ${reference.dimensions_m.map(x => (x * 1000).toFixed(0)).join(" × ")} mm. Authored geometry; published size and mass. Estimated response and box inertia; research camera optics.`;
        sensorNames = message.description.cameras.map((c) => c.robot_id);
        sensorBodies = message.description.cameras.map((c) => world!.bodies[c.body - 1]);
        shown = null; replaying = false;
        if (message.description.site) {
          document.querySelector("h1")!.textContent =
            message.description.site.name;
          document.querySelector("header p")!.textContent =
            "Inspection & model validation";
          camera.position.set(-6, -8, 5);
          controls.target.set(2, 0, 1.4);
        }
        assets = message.description.targets;
        inspectionStatus(null, [true, true], assets);
        state = message.state;
        world.apply(state.bodies, state.time_s);
        guides = new FlightGuides(["Drone", "Rover"]);
        world.scene.add(guides);
        await cameras.capture(
          renderer,
          world.scene,
          sensorBodies,
          state.camera,
        );
        await renderer.waitForGPU();
        send({ type: "ready" });
        loading = false;
        (document.getElementById("pause") as HTMLButtonElement).disabled =
          false;
        (document.getElementById("link") as HTMLInputElement).disabled = false;
        delivered = previous = frames = 0;
        reported = performance.now();
        document.getElementById("status")!.textContent = message.label;
      } else if (message.type === "capture") pending = message;
      else if (message.type === "state") state = message.state;
      else if (message.type === "metrics") {
        delivered = message.count;
        inspectionStatus(message.handover, message.sensor_valid, assets);
        completed = Boolean(message.done);
        document.getElementById("status")!.textContent = message.status;
      } else if (message.type === "saved") {
        completed = true;
        replay.saved(message.captures);
        (document.getElementById("apply-arm") as HTMLButtonElement).disabled = true;
        (document.getElementById("link") as HTMLInputElement).disabled = true;
        (document.getElementById("pause") as HTMLButtonElement).disabled = true;
        document.getElementById("status")!.textContent =
          `Saved ${message.path}`;
      } else if (message.type === "replay") {
        if (!await replay.show(message, sensorNames)) return;
        replaying = true; shown = message.state;
        document.getElementById("handover")!.textContent = "Recorded raw acquisition · mission decisions and delayed observations are retained separately in the run.";
        panel.state(shown, false);
        panel.captureLabel(`REPLAY · capture ${shown.sequence} · ${shown.time_s.toFixed(3)} s · original recorded pixels`);
      } else if (message.type === "error") throw Error(message.message);
    } catch (error) {
      fail(error);
      socket.close();
    }
  };
  socket.onclose = () => {
    (document.getElementById("pause") as HTMLButtonElement).disabled = true;
    (document.getElementById("link") as HTMLInputElement).disabled = true;
    (document.getElementById("apply-arm") as HTMLButtonElement).disabled = true;
    if (!completed && !document.getElementById("error")!.textContent)
      fail(
        "Simulation connection closed. Restart the bridge and reset the mission.",
      );
  };
  socket.onerror = () =>
    fail("Start scripts/run_robotics.py beside the viewer.");
  document.getElementById("focus-robot")!.onclick = () => {
    const pose = (shown ?? state)?.camera_poses[sensorNames.indexOf(panel.selection())];
    if (!pose) return;
    closeUp = true;
    controls.target.fromArray(pose.position_m);
    camera.position.copy(controls.target).add(panel.selection() === "drone"
      ? new T.Vector3(.35, -.38, .24)
      : new T.Vector3(-2, panel.selection() === "arm" ? 2 : -2, 1.5));
  };
  document.getElementById("apply-arm")!.onclick = () => {
    try {
      const control = panel.armValues();
      document.getElementById("error")!.textContent = "";
      send({type: "arm", control});
    } catch (error) { fail(error); }
  };
  document.getElementById("overview")!.onclick = () => {
    closeUp = false;
    camera.position.set(-12, -16, 11);
    controls.target.set(3, 0, 2);
  };
  document.getElementById("equipment")!.onclick = () => {
    closeUp = false;
    camera.position.set(-6, -8, 5);
    controls.target.set(2, 0, 1.4);
  };
  document.getElementById("reset")!.onclick = () => location.reload();
  document.getElementById("pause")!.onclick = () => send({ type: "pause" });
  document.getElementById("link")!.onchange = (e) =>
    send({ type: "link", enabled: (e.target as HTMLInputElement).checked });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) send({ type: "pause" });
  });
  renderer.setAnimationLoop(async (now) => {
    if (
      !world ||
      loading ||
      !state ||
      drawing ||
      document.hidden ||
      now - last < 1000 / 30
    )
      return;
    drawing = true;
    last = now - ((now - last) % (1000 / 30));
    try {
      if (pending) {
        const request = pending;
        pending = null;
        world.apply(request.state.bodies, request.state.time_s);
        const result = await cameras.capture(
          renderer,
          world.scene,
          sensorBodies,
          request.state.camera,
        );
        shown = request.state; panel.state(shown, true);
        for (let i = 0; i < sensorBodies.length; i++)
          (document.getElementById(`camera-${i}`) as HTMLCanvasElement)
            .getContext("2d")!
            .putImageData(
              new ImageData(
                new Uint8ClampedArray(result.data[i * 4]),
                512,
                384,
              ),
              0,
              0,
            );
        send({ type: "camera", id: request.id });
        socket.send(result.packed);
      }
      const displayed = shown ?? state;
      world.apply(displayed.bodies, displayed.time_s);
      if (guides) guides.visible = !replaying && !closeUp;
      if (!replaying) panel.captureLabel(`${completed ? "PAUSED" : "LIVE"} · capture ${displayed.sequence} · ${displayed.time_s.toFixed(3)} s · age ${Math.max(0, (state.time_s - displayed.time_s) * 1000).toFixed(0)} ms`);
      if (!replaying) guides?.update({
        positions: displayed.bodies.positions.slice(0, 2),
        quaternions: displayed.bodies.quaternions.slice(0, 2),
        sequence: Math.floor(displayed.sequence / 10),
      });
      if (view.clientWidth) {
        controls.update();
        renderer.render(world.scene, camera);
        await renderer.waitForGPU();
        frames++;
      }
      if (now - reported > 2000) {
        const seconds = (now - reported) / 1000;
        const fps = frames / seconds,
          hz = (delivered - previous) / seconds;
        document.getElementById("metrics")!.textContent =
          `${displayed.time_s.toFixed(1)} s · ${fps.toFixed(1)} fps · ${hz.toFixed(1)} camera batches/s`;
        send({
          type: "display",
          fps,
          camera_hz: hz,
          width: renderer.domElement.width,
          height: renderer.domElement.height,
          time_s: state.time_s,
        });
        reported = now;
        previous = delivered;
        frames = 0;
      }
    } catch (error) {
      fail(error);
      socket.close();
      renderer.setAnimationLoop(null);
    } finally {
      drawing = false;
    }
  });
  window.addEventListener("pagehide", () => {
    socket.close();
    renderer.setAnimationLoop(null);
    cameras?.dispose();
    controls.dispose();
    if (world) {
      disposeScene(world.scene);
      world.disposeEnvironment();
    }
    renderer.dispose();
  });
}
