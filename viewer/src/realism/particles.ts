import * as T from "three/webgpu";
import { DustCloud } from "./dust-cloud";
import { forestTextures } from "../forest/textures";

export class ParticleView extends T.Group {
  clouds: T.InstancedMesh[];
  dust = new DustCloud();
  constructor() {
    super();
    this.clouds = [
      new T.InstancedMesh(
        new T.PlaneGeometry(0.055, 0.11),
        new T.MeshStandardMaterial({
          map: forestTextures().leaf,
          alphaTest: 0.48,
          side: T.DoubleSide,
          roughness: 1,
        }),
        160,
      ),
      new T.InstancedMesh(
        new T.CylinderGeometry(0.001, 0.001, 0.045, 3),
        new T.MeshStandardMaterial({
          color: 0xb3c8d0,
          roughness: 0.12,
          metalness: 0.1,
        }),
        320,
      ),
    ];
    this.add(this.dust);
    for (const cloud of this.clouds) {
      cloud.instanceMatrix.setUsage(T.DynamicDrawUsage);
      cloud.frustumCulled = false;
      cloud.count = 0;
      this.add(cloud);
    }
  }
  update(points: number[][], kinds: number[], time: number) {
    this.dust.update(points.filter((_,i)=>kinds[i]===0));
    const counts = [0, 0],
      object = new T.Object3D();
    points.forEach((p, i) => {
      const kind = kinds[i];
      if(kind===0)return;
      object.position.fromArray(p);
      object.rotation.set(
        kind === 2 ? Math.PI / 2 : 0,
        kind === 1 ? Math.sin(time * 2 + i) * 0.3 : 0,
        kind === 1 ? i * 2.4 : 0,
      );
      object.updateMatrix();
      this.clouds[kind-1].setMatrixAt(counts[kind-1]++, object.matrix);
    });
    this.clouds.forEach((cloud, i) => {
      cloud.count = counts[i];
      cloud.instanceMatrix.needsUpdate = true;
    });
  }
}
