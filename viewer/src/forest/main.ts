import { panel, el } from "../dashboard/panel";
import { playback } from "../dashboard/playback";
import { disposeScene } from "../dispose-scene";
import * as T from "three/webgpu";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { buildForest } from "./trees";
import { scannedStumps } from "./scanned-models";
import { forestEnvironment } from "./environment";
import { prepareForestTextures } from "./scanned-materials";
import { sunrise } from "./sunrise";
import { forestGround } from "./ground";
import type { Episode } from "../types";
export async function start() {
  await prepareForestTextures();
  T.Object3D.DEFAULT_UP.set(0, 0, 1);
  const response = await fetch("/data/forest-held-out.json");
  if (!response.ok) throw Error("Forest episode unavailable");
  const episode: Episode = await response.json();
  const ui = panel();
  ui.setup([{id: 'drone', label: 'Drone', signals: [['height', 'Recorded body height (m)']]}], [{id: 'drone', label: 'Original recorded RGB · original scene', width: episode.frameWidth, height: episode.frameHeight}]);
  const atlas = new Image(); atlas.src = `/data/${episode.atlas}`; await atlas.decode();
  const samples = episode.records.map(r => ({time_s: r.time_s, robots: {drone: {position: r.position, yaw: Math.atan2(2 * (r.quaternion[0] * r.quaternion[3] + r.quaternion[1] * r.quaternion[2]), 1 - 2 * (r.quaternion[2] ** 2 + r.quaternion[3] ** 2)) * 180 / Math.PI, signals: {height: r.position[2]}}}})); ui.history(samples);
  el("pose-title").textContent = "Body pose";
  el('source-controls').hidden = false;
  el('source-controls').innerHTML = '<h3>Forest rendering</h3><button id="drone-camera">Drone camera</button><button id="cabin">Cabin clearing</button><button id="save">Save 1536 × 1152 render</button>';
  el('handover').textContent = 'Scene is a detailed re-render at recorded poses. Sensor panel retains original source pixels. This renderer was not used to train the policy.';
  const scene = new T.Scene();
  scene.background = new T.Color(0xb7c7ce);
  scene.fog = new T.FogExp2(0xb7c7be, 0.025);
  const wind = episode.scene.environment?.wind_m_s;
  if (!wind) throw Error("Forest environment requires wind metadata");
  const forest = buildForest(wind);
  scene.add(forest.root);
  forestGround(forest.root);
  await scannedStumps(forest.root);
  sunrise(scene);
  const renderer = new T.WebGPURenderer({ antialias: true });
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = T.PCFSoftShadowMap;
  renderer.toneMapping = T.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  await renderer.init();
  const environment = await forestEnvironment(renderer, scene);
  const view = document.getElementById("view")!;
  view.prepend(renderer.domElement);
  const camera = new T.PerspectiveCamera(63, 4 / 3, 0.035, 80);
  camera.up.set(0, 0, 1);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.enabled = false;
  controls.maxPolarAngle = Math.PI * 0.49;
  let index = 0, orbit = false, capturing = false;
  const clock = playback(episode.records.map(r => r.time_s), i => {
    index = i; ui.state(samples[i], false);
    const w = episode.frameWidth, h = episode.frameHeight;
    el<HTMLCanvasElement>('camera-0').getContext('2d')!.drawImage(atlas, (i % episode.atlasColumns) * w, Math.floor(i / episode.atlasColumns) * h, w, h, 0, 0, w, h);
    ui.captureLabel(`REPLAY · ${episode.records[i].time_s.toFixed(3)} s · original RGB; detailed scene re-render`);
  });
  function pose() {
    const r = episode.records[index],
      q = new T.Quaternion(
        r.quaternion[1],
        r.quaternion[2],
        r.quaternion[3],
        r.quaternion[0],
      );
    const f = new T.Vector3(1, 0, 0).applyQuaternion(q),
      left = new T.Vector3(0, 1, 0).applyQuaternion(q),
      up = new T.Vector3(0, 0, 1).applyQuaternion(q);
    camera.position
      .fromArray(r.position)
      .addScaledVector(f, 0.035)
      .addScaledVector(up, 0.012);
    camera.quaternion.setFromRotationMatrix(
      new T.Matrix4().makeBasis(left.negate(), up, f.negate()),
    );
  }
  function resize() {
    if (!view.clientWidth || !view.clientHeight) return;
    renderer.setSize(view.clientWidth, view.clientHeight);
    camera.aspect = view.clientWidth / view.clientHeight;
    camera.updateProjectionMatrix();
  }
  const observer = new ResizeObserver(resize); observer.observe(view);
  resize();
  pose();
  document.getElementById("drone-camera")!.onclick = () => {
    orbit = false;
    controls.enabled = false;
    pose();
    document.getElementById("drone-camera")!.classList.add("active");
    document.getElementById("overview")!.classList.remove("active");
  };
  document.getElementById("overview")!.onclick = () => {
    orbit = true;
    controls.enabled = true;
    camera.position.set(-5, -5, 2.5);
    controls.target.set(1, 0, 2);
    document.getElementById("overview")!.classList.add("active");
    document.getElementById("drone-camera")!.classList.remove("active");
  };
  document.getElementById("cabin")!.onclick = () => {
    orbit = true;
    controls.enabled = true;
    camera.position.set(1.5, -12, 2.1);
    controls.target.set(5, -7, 1.5);
  };
  el('focus-robot').onclick = () => el('drone-camera').click();
  document.getElementById("save")!.onclick = async () => {
    clock.seek(index);
    const button = document.getElementById("save") as HTMLButtonElement;
    button.disabled = true;
    capturing = true;
    document.getElementById("error")!.textContent = "";
    try {
      camera.aspect = 4 / 3;
      camera.updateProjectionMatrix();
      renderer.setPixelRatio(1);
      renderer.setSize(1536, 1152, false);
      await renderer.renderAsync(scene, camera);
      const rgb = renderer.domElement.toDataURL("image/png");
      const response = await fetch("/__forest-capture", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          rgb,
          time_s: episode.records[index].time_s,
          position: camera.position.toArray(),
          quaternion: camera.quaternion.toArray(),
          view: orbit ? "observer" : "drone",
          scene: episode.scene.identity,
            resolution: [1536,1152], vertical_fov_deg: 63, quaternion_order: "xyzw",
            source: "webgpu_quality_rerender", training_consumed: false,
        }),
      });
      if (!response.ok) throw Error(await response.text());
      document.getElementById("status")!.textContent =
        "Saved actual 1536 × 1152 GPU render and camera metadata.";
    } catch (e) {
      document.getElementById("error")!.textContent = String(e);
    } finally {
      renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
      resize();
      capturing = false;
      button.disabled = false;
    }
  };
  let lastDraw = 0;
  el("status").textContent = "Recorded poses and original sensor images. Detailed scene rendering did not train the policy.";
  let frames = 0,
    started = performance.now();
  renderer.setAnimationLoop((now: number) => {
    if (capturing || document.hidden || !view.clientWidth || now - lastDraw < 1000 / 30) return;
    lastDraw = now - ((now - lastDraw) % (1000 / 30));
    const r = episode.records[index];
    if (!orbit) pose();
    forest.update(r.time_s, r.wind_m_s ?? [0, 0, 0]);
    if (orbit) controls.update();
    renderer.render(scene, camera);
    frames++;
    if (frames % 90 === 0)
      document.getElementById("metrics")!.textContent =
        `${((frames * 1000) / (performance.now() - started)).toFixed(0)} display FPS · ${forest.leaves.count.toLocaleString()} leaves · Recorded poses, new renderer. Earlier policy results used the original camera.`;
  });
  window.addEventListener('pagehide', () => {
    observer.disconnect(); renderer.setAnimationLoop(null); controls.dispose(); disposeScene(scene); environment(); renderer.dispose();
  }, {once: true});
}
