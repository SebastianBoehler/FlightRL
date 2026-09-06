import * as T from "three/webgpu";
import { instancedBufferAttribute, vec2 } from "three/tsl";
import { random } from "../forest/textures";

/** Stochastic coverage gives dust soft silhouettes without allocating rigid bodies. */
export class DustCloud extends T.Sprite {
  count = 0;
  positions: T.InstancedBufferAttribute;
  constructor() {
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 64;
    const context = canvas.getContext("2d")!,
      pixels = context.createImageData(64, 64),
      rng = random(828);
    for (let y = 0; y < 64; y++)
      for (let x = 0; x < 64; x++) {
        const radius = Math.hypot((x - 31.5) / 32, (y - 31.5) / 32),
          i = (y * 64 + x) * 4;
        pixels.data.set(
          [
            255,
            255,
            255,
            Math.round(
              255 * Math.max(0, 1 - radius) ** 2 * (0.65 + rng() * 0.35),
            ),
          ],
          i,
        );
      }
    context.putImageData(pixels, 0, 0);
    const positions = new T.InstancedBufferAttribute(
      new Float32Array(1024 * 3),
      3,
    ).setUsage(T.DynamicDrawUsage);
    const material = new T.SpriteNodeMaterial({
      color: 0xb69c79,
      map: new T.CanvasTexture(canvas),
      alphaHash: true,
      opacity: 0.55,
      depthWrite: true,
    });
    material.positionNode = instancedBufferAttribute(positions);
    material.scaleNode = vec2(0.18);
    super(material);
    this.positions = positions;
    this.frustumCulled = false;
  }
  update(points: number[][]) {
    const positions = this.positions;
    points.forEach((p, i) => positions.setXYZ(i, p[0], p[1], p[2]));
    positions.needsUpdate = true;
    this.count = points.length;
  }
}
