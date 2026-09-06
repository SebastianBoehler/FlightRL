import * as T from "three/webgpu";
import { mrt, output, positionView, vec4 } from "three/tsl";
import type { WorldState } from "./types";

export const SIZES = [
  [256, 192],
  [64, 48],
] as const;
export function cameraPose(camera: T.Camera, p: number[], q: number[], offset = [0.035, 0, 0.012]) {
  const rotation = new T.Quaternion().fromArray(q);
  const f = new T.Vector3(1, 0, 0).applyQuaternion(rotation),
    l = new T.Vector3(0, 1, 0).applyQuaternion(rotation);
  const up = new T.Vector3(0, 0, 1).applyQuaternion(rotation);
  camera.position
    .fromArray(p)
    .addScaledVector(f, offset[0])
    .addScaledVector(l, offset[1])
    .addScaledVector(up, offset[2]);
  camera.quaternion.setFromRotationMatrix(
    new T.Matrix4().makeBasis(l.negate(), up, f.negate()),
  );
}

/** Same opaque/cutout geometry pass writes color and metric ray distance. */
export class Cameras {
  camera = new T.PerspectiveCamera(63, 4 / 3, 0.01, 80);
  targets: T.RenderTarget[][];
  outputs = mrt({
    output,
    distance: vec4(positionView.length().min(8), 0, 0, 1),
  });
  constructor(
    count = 3,
    sizes: ReadonlyArray<readonly [number, number]> = SIZES,
    private offsets?: number[][],
  ) {
    this.targets = Array.from({ length: count }, () =>
      sizes.map(([w, h]) => {
        const target = new T.RenderTarget(w, h, {
          count: 2,
          depthBuffer: true,
          samples: 0,
        });
        target.textures[0].name = "output";
        target.textures[0].type = T.UnsignedByteType;
        target.textures[0].colorSpace = T.SRGBColorSpace;
        target.textures[1].name = "distance";
        target.textures[1].type = T.FloatType;
        target.textures[1].format = T.RedFormat;
        target.textures[1].colorSpace = T.NoColorSpace;
        return target;
      }),
    );

    this.camera.up.set(0, 0, 1);
  }

  async capture(
    renderer: T.WebGPURenderer,
    scene: T.Scene,
    bodies: T.Object3D[],
    state: Pick<WorldState, "positions" | "quaternions">,
  ) {
    const old = renderer.getRenderTarget(),
      oldMrt = renderer.getMRT();
    const data: Array<Uint8Array | Float32Array> = [];
    const background = scene.background;
    try {
      renderer.setMRT(this.outputs);
      // Background must write max range, so draw sky normally then repair only zero clear depth.
      for (let i = 0; i < this.targets.length; i++) {
        cameraPose(this.camera, state.positions[i], state.quaternions[i], this.offsets?.[i]);
        bodies[i].visible = false;
        for (let level = 0; level < this.targets[i].length; level++) {
          const target = this.targets[i][level];
          renderer.setRenderTarget(target);
          renderer.render(scene, this.camera);
        }
        bodies[i].visible = true;
      }
      const buffers = await Promise.all(
        this.targets.flatMap((levels) =>
          levels.flatMap((target) => [
            renderer.readRenderTargetPixelsAsync(
              target,
              0,
              0,
              target.width,
              target.height,
              0,
            ),
            renderer.readRenderTargetPixelsAsync(
              target,
              0,
              0,
              target.width,
              target.height,
              1,
            ),
          ]),
        ),
      );
      for (let i = 0; i < buffers.length; i++) {
        const buffer = buffers[i];
        if (i % 2 === 1) {
          const depth = buffer as Float32Array;
          for (let k = 0; k < depth.length; k++)
            if (depth[k] === 0) depth[k] = 8;
        }
        data.push(buffer as Uint8Array | Float32Array);
      }
      const packed = new Uint8Array(data.reduce((n, a) => n + a.byteLength, 0));
      let offset = 0;
      for (const a of data) {
        packed.set(
          new Uint8Array(a.buffer, a.byteOffset, a.byteLength),
          offset,
        );
        offset += a.byteLength;
      }
      return { packed, data };
    } finally {
      bodies.forEach((b) => (b.visible = true));
      scene.background = background;
      renderer.setMRT(oldMrt);
      renderer.setRenderTarget(old);
    }
  }
  dispose() {
    this.targets.flat().forEach((t) => t.dispose());
  }
}
