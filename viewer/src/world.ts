import * as THREE from "three/webgpu";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { Episode, Frame } from "./types";
import { roomView } from "./room-view";
import { buildForest, type ForestAssets } from "./forest/trees";
import { ForestFeed } from "./forest/feed";
import { prepareForestTextures } from "./forest/scanned-materials";
import { sunrise } from "./forest/sunrise";
import { forestGround } from "./forest/ground";
import { DustView } from "./dust-view";
import { AirflowView } from "./airflow-view";
import { buildDrone } from "./drone";
import { plantDetails, setCutaway } from "./plant-details";
import { disposeScene } from "./dispose-scene";

const material = (color: number, metalness = 0, roughness = 0.65) =>
  new THREE.MeshStandardMaterial({ color, metalness, roughness });
export class World {
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(43, 1, 0.03, 100);
  renderer = new THREE.WebGPURenderer({ antialias: true });
  controls: OrbitControls;
  root = new THREE.Group();
  drone = new THREE.Group();
  onboard = new THREE.PerspectiveCamera(63, 4 / 3, 0.05, 1.4);
  frustum: THREE.CameraHelper;
  route = new THREE.Line(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: 0x59d7c5 }),
  );
  markers: THREE.Mesh[] = [];
  airflow = new AirflowView();
  dust = new DustView();
  atlasCanvas = document.createElement("canvas");
  texture: THREE.CanvasTexture;
  mode = "Overview";
  current: Frame | null = null;
  episode: Episode | null = null;
  forest: ForestAssets | null = null;
  feed = new ForestFeed();
  habitatLights: THREE.Object3D[] = [];
  constructor(private container: HTMLElement, private rerenderFeed = true) {
    THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
    this.camera.up.set(0, 0, 1);
    this.scene.background = new THREE.Color(0x10171c);
    this.scene.fog = new THREE.Fog(0x10171c, 22, 45);
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;
    container.append(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.minDistance = 1;
    this.controls.maxDistance = 22;
    this.controls.maxPolarAngle = Math.PI * 0.48;
    this.scene.add(new THREE.HemisphereLight(0xc9e0f5, 0x28363c, 2.5));
    const light = new THREE.DirectionalLight(0xffefd9, 4);
    light.position.set(-2, -4, 10);
    light.castShadow = true;
    light.shadow.mapSize.set(2048, 2048);
    light.shadow.camera.left = -8;
    light.shadow.camera.right = 8;
    light.shadow.camera.top = 8;
    light.shadow.camera.bottom = -8;
    this.dust.visible = false;
    this.scene.add(this.dust, this.airflow);
    this.scene.add(light, this.root, this.drone, this.onboard, this.route);
    this.frustum = new THREE.CameraHelper(this.onboard);
    this.frustum.setColors(
      new THREE.Color(0x59d7c5),
      new THREE.Color(0x59d7c5),
      new THREE.Color(0x59d7c5),
      new THREE.Color(0x29433e),
      new THREE.Color(0x29433e),
    );
    this.scene.add(this.frustum);
    this.atlasCanvas.width = 64;
    this.atlasCanvas.height = 48;
    this.atlasCanvas.getContext("2d");
    this.texture = new THREE.CanvasTexture(this.atlasCanvas);
    this.texture.magFilter = THREE.NearestFilter;
    this.texture.colorSpace = THREE.SRGBColorSpace;
    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(
        (2 * 0.85 * Math.tan((63 * Math.PI) / 360) * 4) / 3,
        2 * 0.85 * Math.tan((63 * Math.PI) / 360),
      ),
      new THREE.MeshBasicMaterial({
        map: this.texture,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.8,
      }),
    );
    plane.position.z = -0.85;
    this.onboard.add(plane);
    this.route.visible = false;
    buildDrone(this);
    this.overview();
    const observer = new ResizeObserver(() => this.resize()); observer.observe(container);
    window.addEventListener("pagehide", () => {
      observer.disconnect(); this.renderer.setAnimationLoop(null); this.controls.dispose();
      this.feed.reset(); disposeScene(this.scene); this.renderer.dispose();
    }, {once: true});
  }
  async start() {
    await prepareForestTextures();
    await this.renderer.init();
    await this.feed.start();
    if (
      !(this.renderer.backend as { isWebGPUBackend?: boolean }).isWebGPUBackend
    )
      throw new Error("This viewer requires WebGPU.");
    let drawing = false, lastDraw = 0;
    this.renderer.setAnimationLoop(async (now: number) => {
      if (drawing || document.hidden || now - lastDraw < 1000 / 30) return;
      drawing = true;
      lastDraw = now;
      try {
        this.controls.update();
        this.feed.flush(this.renderer);
        if (this.container.clientWidth) this.renderer.render(this.scene, this.camera);
        await this.renderer.waitForGPU();
      } catch (error) {
        const el = document.getElementById("error")!;
        el.hidden = false; el.textContent = String(error);
      } finally { drawing = false; }
    });
    this.resize();
  }
  box(
    size: number[],
    position: number[],
    mat: THREE.Material,
    parent = this.root,
  ) {
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(...(size as [number, number, number])),
      mat,
    );
    mesh.position.set(...(position as [number, number, number]));
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    parent.add(mesh);
    return mesh;
  }
  load(episode: Episode) {
    this.episode = episode;
    disposeScene(this.root);
    this.root.clear();
    this.route.geometry.dispose();
    this.route.geometry = new THREE.BufferGeometry().setFromPoints(
      episode.records.map((r) => new THREE.Vector3(...r.position)),
    );
    this.route.geometry.setDrawRange(0, 0);
    this.atlasCanvas.width = episode.frameWidth;
    this.atlasCanvas.height = episode.frameHeight;
    this.texture.dispose();
    this.texture.needsUpdate = true;
    this.markers = [];
    const room = episode.scene.room;
    const forest = episode.scene.environment?.surface_style === "forest";
    this.habitatLights.forEach(light => { disposeScene(light); this.scene.remove(light); });
    this.habitatLights = [];
    this.scene.children.filter(o => o instanceof THREE.Light).forEach(o => o.visible = !forest);
    this.forest = null;
    if (forest) {
      const wind = episode.scene.environment?.wind_m_s;
      if (!wind) throw Error("Forest wind metadata missing");
      const previous = new Set(this.scene.children);
      sunrise(this.scene);
      this.habitatLights = this.scene.children.filter(o => !previous.has(o));
      this.forest = buildForest(wind);
      forestGround(this.forest.root);
      this.root.add(this.forest.root);
    } else { this.setTheme(document.documentElement.dataset.theme === "light"); roomView(this, room, false); }
    for (const b of episode.scene.boxes) {
      if (forest) continue;
      this.box(
        [b[1] - b[0], b[3] - b[2], b[5] - b[4]],
        [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2],
        material(b[4] > 2 ? 0xa6adae : b[5] > 3 ? 0x879293 : 0x405e69, 0.65),
      );
      if (b[4] > 2 || b[5] > 3) continue;
      for (let z = 0.3; z < b[5] - 0.1; z += 0.12)
        this.box(
          [0.006, b[3] - b[2] - 0.12, 0.025],
          [b[0] - 0.004, (b[2] + b[3]) / 2, z],
          material(0x1b2b31, 0.5),
        );
      this.box(
        [b[1] - b[0] + 0.1, b[3] - b[2] + 0.1, 0.01],
        [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, 0.02],
        material(0xd5ad51),
      );
    }
    if (!forest) plantDetails(this.root, episode.scene.boxes);
    for (const p of episode.scene.panels) {
      const mesh = new THREE.Mesh(
        new THREE.PlaneGeometry(p[9] * 2, p[10] * 2),
        material(
          new THREE.Color(p[11] / 255, p[12] / 255, p[13] / 255).getHex(),
          0.25,
        ),
      );
      const u = new THREE.Vector3(
          ...(p.slice(3, 6) as [number, number, number]),
        ),
        v = new THREE.Vector3(...(p.slice(6, 9) as [number, number, number]));
      mesh.quaternion.setFromRotationMatrix(
        new THREE.Matrix4().makeBasis(u, v, u.clone().cross(v)),
      );
      mesh.position.set(...(p.slice(0, 3) as [number, number, number]));
      this.root.add(mesh);
      this.markers.push(mesh);
      const border = new THREE.LineSegments(
        new THREE.EdgesGeometry(
          new THREE.PlaneGeometry(p[9] * 2 + 0.08, p[10] * 2 + 0.08),
        ),
        new THREE.LineBasicMaterial({ color: 0xd5e0e2 }),
      );
      border.position.copy(mesh.position);
      border.quaternion.copy(mesh.quaternion);
      this.root.add(border);
    }
    if (forest) {
      this.root.traverse(object => object.layers.enable(1));
      this.habitatLights.forEach(object => object.layers.enable(1));
    }
    this.overview();
  }
  update(frame: Frame, index: number, atlas: HTMLImageElement) {
    this.current = frame;
    this.forest?.update(frame.time_s, frame.wind_m_s ?? [0, 0, 0]);
    this.airflow.update(frame);
    this.dust.visible = !!(
      frame.particles?.length || frame.settled_particles?.length
    );
    this.dust.update([
      ...(frame.particles ?? []),
      ...(frame.settled_particles ?? []),
    ]);
    const p = new THREE.Vector3(
      ...(frame.position as [number, number, number]),
    );
    const q = new THREE.Quaternion(
      frame.quaternion[1],
      frame.quaternion[2],
      frame.quaternion[3],
      frame.quaternion[0],
    );
    this.drone.position.copy(p);
    this.drone.quaternion.copy(q);
    const forward = new THREE.Vector3(1, 0, 0).applyQuaternion(q),
      left = new THREE.Vector3(0, 1, 0).applyQuaternion(q),
      up = new THREE.Vector3(0, 0, 1).applyQuaternion(q);
    this.onboard.position
      .copy(p)
      .addScaledVector(forward, 0.035)
      .addScaledVector(up, 0.012);
    this.onboard.quaternion.setFromRotationMatrix(
      new THREE.Matrix4().makeBasis(left.negate(), up, forward.negate()),
    );
    this.onboard.updateMatrixWorld();
    if (this.forest && this.rerenderFeed) {
      const currentEpisode = this.episode;
      void this.feed.render(this.scene, this.onboard, () => !!this.current?.connected, () => this.episode === currentEpisode, canvas => {
          if (this.atlasCanvas.width !== 512) {
            this.texture.dispose();
            this.atlasCanvas.width = 512; this.atlasCanvas.height = 384;
          }
          this.atlasCanvas.getContext("2d")!.drawImage(canvas, 0, 0);
          this.texture.needsUpdate = true;
          document.getElementById("feed-caption")!.textContent = `Live detailed re-render · frame ${frame.time_s.toFixed(1)} s · not training observations`;
        })
        .catch(error => { const el = document.getElementById("error")!; el.hidden = false; el.textContent = String(error); });
    }
    this.frustum.update();
    if (!this.forest || !this.rerenderFeed) {
    const context = this.atlasCanvas.getContext("2d")!;
    const columns = this.episode!.atlasColumns;
    const w = this.episode!.frameWidth,
      h = this.episode!.frameHeight;
    context.drawImage(
      atlas,
      (index % columns) * w,
      Math.floor(index / columns) * h,
      w,
      h,
      0,
      0,
      w,
      h,
    );
    this.texture.needsUpdate = true;
    }
    this.route.visible = index > 0;
    this.route.geometry.setDrawRange(0, index + 1);
    this.markers.forEach((m, i) => {
      (m.material as THREE.MeshStandardMaterial).emissive.setHex(
        frame.truth_inspected.includes(101 + i) ? 0x284536 : 0,
      );
    });
    setCutaway(this.root, this.markers, this.mode === "Camera pose");
    if (this.mode === "Follow drone") {
      this.camera.position.copy(p).add(new THREE.Vector3(-3, -3, 2));
      this.controls.target.copy(p);
    }
    if (this.mode === "Camera pose") {
      this.camera.position
        .copy(p)
        .add(new THREE.Vector3(-1.7, -1.4, 1.2).applyQuaternion(q));
      this.controls.target
        .copy(this.onboard.position)
        .add(new THREE.Vector3(0.4, 0, 0).applyQuaternion(q));
    }
  }
  overview() {
    const room = this.episode?.scene.room;
    const center = room ? (room[0] + room[1]) / 2 : 0;
    this.camera.position.set(center - 10, -13, 12);
    this.controls.target.set(center, 0, 0.5);
    this.controls.update();
  }
  setTheme(light: boolean) {
    this.scene.background = new THREE.Color(light ? 0xe3e8ec : 0x10171c);
    this.scene.fog = new THREE.Fog(light ? 0xe3e8ec : 0x10171c, 22, 45);
  }
  resize() {
    const w = this.container.clientWidth,
      h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }
}
