import * as T from "three/webgpu";
import type { WorldState } from "./types";

/** Observer annotations live on layer 1; sensor cameras only render layer 0. */
export class FlightGuides extends T.Group {
  private lastSequence = -5;
  private tracks;

  constructor(labels = ["Drone 1", "Drone 2", "Drone 3"]) {
    super();
    this.tracks = labels.map((name, i) => {
      const color = [0xd05c43, 0x3e8bc3, 0x6faa62][i];
      const positions = new Float32Array(600 * 3);
      const geometry = new T.BufferGeometry();
      geometry.setAttribute(
        "position",
        new T.BufferAttribute(positions, 3).setUsage(T.DynamicDrawUsage),
      );
      geometry.setDrawRange(0, 0);
      const line = new T.Line(
        geometry,
        new T.LineBasicMaterial({
          color,
          transparent: true,
          opacity: 0.8,
          depthTest: false,
          depthWrite: false,
        }),
      );
      line.frustumCulled = false;
      const arrow = new T.ArrowHelper(
        new T.Vector3(1, 0, 0),
        new T.Vector3(),
        0.8,
        color,
        0.18,
        0.12,
      );
      const ring = new T.Mesh(
        new T.TorusGeometry(0.2, 0.012, 5, 24),
        new T.MeshBasicMaterial({ color, depthTest: false, depthWrite: false }),
      );
      const canvas = document.createElement("canvas");
      canvas.width = 128;
      canvas.height = 64;
      const context = canvas.getContext("2d")!;
      context.fillStyle = "#112018dd";
      context.fillRect(0, 0, 128, 64);
      context.fillStyle = `#${color.toString(16).padStart(6, "0")}`;
      context.font = "bold 28px system-ui";
      context.textAlign = "center";
      context.fillText(name, 64, 42);
      const texture = new T.CanvasTexture(canvas);
      texture.colorSpace = T.SRGBColorSpace;
      const label = new T.Sprite(
        new T.SpriteMaterial({
          map: texture,
          depthTest: false,
          depthWrite: false,
        }),
      );
      label.scale.set(0.85, 0.425, 1);
      this.add(line, arrow, ring, label);
      return { positions, geometry, arrow, ring, label, count: 0 };
    });

    this.traverse((object) => {
      object.layers.set(1);
      object.renderOrder = 1000;
      if (object instanceof T.Mesh || object instanceof T.Line) {
        const materials = Array.isArray(object.material)
          ? object.material
          : [object.material];
        materials.forEach((material) => {
          material.depthTest = false;
          material.depthWrite = false;
        });
      }
    });
  }

  update(state: Pick<WorldState, "positions" | "quaternions" | "sequence">) {
    const append = state.sequence >= this.lastSequence + 5;
    this.tracks.forEach((track, i) => {
      const position = new T.Vector3().fromArray(state.positions[i]);
      const rotation = new T.Quaternion().fromArray(state.quaternions[i]);
      track.arrow.position.copy(position);
      track.arrow.setDirection(
        new T.Vector3(1, 0, 0).applyQuaternion(rotation),
      );
      track.ring.position.copy(position);
      track.ring.quaternion.copy(rotation);
      track.label.position.copy(position).add(new T.Vector3(0, 0, 0.5));
      if (append) {
        if (track.count === 600) track.positions.copyWithin(0, 3);
        else track.count++;
        position.toArray(track.positions, (track.count - 1) * 3);
        track.geometry.attributes.position.needsUpdate = true;
        track.geometry.setDrawRange(0, track.count);
      }
    });
    if (append) this.lastSequence = state.sequence;
  }
}
