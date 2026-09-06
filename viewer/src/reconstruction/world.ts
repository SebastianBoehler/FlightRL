import * as T from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { MapData, Point } from "./types";
export class MapWorld {
  private observer: ResizeObserver;
  private renderer = new T.WebGLRenderer({ antialias: true });
  private scene = new T.Scene();
  private camera = new T.PerspectiveCamera(48, 1, 0.01, 500);
  private controls: OrbitControls;
  private cloud: T.Points;
  private trajectory: T.LineSegments;
  private reference: T.LineSegments;
  private trajectoryCounts: number[] = [];
  private referenceCounts: number[] = [];
  private marker = new T.Mesh(
    new T.SphereGeometry(0.065),
    new T.MeshBasicMaterial({ color: 0x57dec1 }),
  );
  constructor(private host: HTMLElement) {
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    host.append(this.renderer.domElement);
    this.scene.background = new T.Color("#0b141a");
    this.camera.position.set(-7, -5, -5);
    this.camera.up.set(0, -1, 0);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0, 3);
    this.controls.update();
    this.cloud = new T.Points(
      new T.BufferGeometry(),
      new T.PointsMaterial({ size: 0.028, vertexColors: true }),
    );
    this.trajectory = new T.LineSegments(
      new T.BufferGeometry(),
      new T.LineBasicMaterial({ color: 0x57dec1 }),
    );
    this.reference = new T.LineSegments(
      new T.BufferGeometry(),
      new T.LineBasicMaterial({ color: 0xf0ab69 }),
    );
    this.scene.add(this.cloud, this.trajectory, this.reference, this.marker);
    this.observer = new ResizeObserver(() => {
      if (!host.clientWidth || !host.clientHeight) return;
      this.renderer.setSize(host.clientWidth, host.clientHeight);
      this.camera.aspect = host.clientWidth / host.clientHeight;
      this.camera.updateProjectionMatrix();
    }); this.observer.observe(host);
    this.renderer.setAnimationLoop(() => {
      if (!host.clientWidth || document.hidden) return;
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }
  dispose() {
    this.observer.disconnect(); this.renderer.setAnimationLoop(null); this.controls.dispose();
    for (const item of [this.cloud, this.trajectory, this.reference, this.marker]) { item.geometry.dispose(); (item.material as T.Material).dispose(); }
    this.renderer.dispose();
  }
  focus(map: MapData, frame: number) {
    const pose = frame > 0 ? map.poses[frame - 1] : null;
    if (!pose) return;
    const offset = this.camera.position.clone().sub(this.controls.target);
    this.controls.target.fromArray(pose); this.camera.position.copy(this.controls.target).add(offset); this.controls.update();
  }
  load(map: MapData) {
    const set = (line: T.LineSegments, points: (Point | null)[]) => {
      const segments: number[] = [], counts = [0];
      for (let i = 0; i < points.length; i++) {
        const a = i > 0 ? points[i - 1] : null, b = points[i];
        // Missing tracking must remain a gap, including after seeking backwards.
        if (a && b) segments.push(...a, ...b);
        counts.push(segments.length / 3);
      }
      line.geometry.dispose();
      line.geometry = new T.BufferGeometry();
      line.geometry.setAttribute("position", new T.Float32BufferAttribute(segments, 3));
      line.geometry.setDrawRange(0, 0);
      return counts;
    };
    this.trajectoryCounts = set(this.trajectory, map.poses);
    this.referenceCounts = set(this.reference, map.truth);
    this.cloud.geometry.dispose();
    this.cloud.geometry = new T.BufferGeometry();
    this.cloud.geometry.setAttribute(
      "position",
      new T.Float32BufferAttribute(
        map.points.flatMap((p) => p[0]),
        3,
      ),
    );
    this.cloud.geometry.setAttribute(
      "color",
      new T.Float32BufferAttribute(
        map.points.flatMap((p) => p[1].map((c) => c / 255)),
        3,
      ),
    );
    this.cloud.geometry.setDrawRange(0, 0);
    if (map.points.length) {
      this.cloud.geometry.computeBoundingSphere();
      const sphere = this.cloud.geometry.boundingSphere!;
      const radius = Math.max(sphere.radius, 0.001);
      this.controls.target.copy(sphere.center);
      this.camera.position
        .copy(sphere.center)
        .add(
          new T.Vector3(-1, -0.6, -1).normalize().multiplyScalar(radius * 3),
        );
      this.camera.near = radius / 1000;
      this.camera.far = radius * 100;
      this.camera.updateProjectionMatrix();
      this.controls.update();
      (this.cloud.material as T.PointsMaterial).size = radius * 0.006;
      this.marker.scale.setScalar((radius * 0.025) / 0.065);
    }
  }
  update(map: MapData, frame: number, showTruth: boolean) {
    const count = map.points.findIndex((p) => p[2] > frame);
    this.cloud.geometry.setDrawRange(0, count < 0 ? map.points.length : count);
    const pose = frame > 0 ? map.poses[frame - 1] : null;
    this.marker.visible = !!pose;
    if (pose) this.marker.position.fromArray(pose);
    this.trajectory.geometry.setDrawRange(0, this.trajectoryCounts[frame]);
    this.reference.geometry.setDrawRange(0, this.referenceCounts[frame]);
    this.reference.visible = showTruth && map.mode === "rgbd";
    return count < 0 ? map.points.length : count;
  }
}
