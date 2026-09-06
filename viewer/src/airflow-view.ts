import * as THREE from "three/webgpu";
import type { Frame } from "./types";

/** Diagnostic glyphs only; the sensor renderer never receives these overlays. */
export class AirflowView extends THREE.Group {
  private arrows: THREE.ArrowHelper[] = [];
  private acceleration = new THREE.ArrowHelper(
    new THREE.Vector3(0, 0, 1),
    new THREE.Vector3(),
    1,
    0xffad66,
  );
  constructor() {
    super();
    this.add(this.acceleration);
  }
  update(frame: Frame) {
    const radius = 1.5;
    const samples = (frame.airflow_samples ?? []).filter(row =>
      Math.hypot(row[0] - frame.position[0], row[1] - frame.position[1], row[2] - frame.position[2]) < radius,
    );
    while (this.arrows.length < samples.length) {
      const arrow = new THREE.ArrowHelper(
        new THREE.Vector3(0, 0, 1),
        new THREE.Vector3(),
        1,
        0x63cfe3,
      );
      this.arrows.push(arrow);
      this.add(arrow);
    }
    this.arrows.forEach((arrow, index) => {
      const row = samples[index];
      arrow.visible = !!row;
      if (!row) return;
      const velocity = new THREE.Vector3(row[3], row[4], row[5]);
      const speed = velocity.length();
      arrow.visible = speed > 0.04;
      if (!arrow.visible) return;
      arrow.position.set(row[0], row[1], row[2]);
      const distance = arrow.position.distanceTo(new THREE.Vector3().fromArray(frame.position));
      const opacity = Math.min(1, (radius - distance) / 0.35);
      for (const part of [arrow.line, arrow.cone]) {
        const material = part.material as THREE.Material;
        material.transparent = true;
        material.opacity = opacity;
      }
      arrow.setDirection(velocity.normalize());
      arrow.setLength(Math.min(0.8, speed * 0.5), 0.08, 0.04);
    });
    const a = new THREE.Vector3(
      ...((frame.gust_m_s2 ?? [0, 0, 0]) as [number, number, number]),
    );
    this.acceleration.visible = a.length() > 0.01;
    if (this.acceleration.visible) {
      this.acceleration.position.fromArray(frame.position);
      this.acceleration.setLength(Math.min(1.5, a.length() * 2), 0.12, 0.07);
      this.acceleration.setDirection(a.normalize());
    }
  }
}
