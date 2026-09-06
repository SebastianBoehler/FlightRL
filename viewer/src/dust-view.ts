import * as THREE from "three/webgpu";

/** Enlarged tracer glyphs, not physical grain diameters or sensor particles. */
export class DustView extends THREE.InstancedMesh {
  private transform = new THREE.Matrix4();
  constructor() {
    super(
      new THREE.OctahedronGeometry(0.007),
      new THREE.MeshBasicMaterial({ color: 0xf0dca6 }),
      8192,
    );
    this.count = 0;
    this.frustumCulled = false;
    this.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  }
  update(points: number[][]) {
    this.count = points.length;
    points.forEach((point, index) => {
      this.transform.makeTranslation(point[0], point[1], point[2]);
      this.setMatrixAt(index, this.transform);
    });
    this.instanceMatrix.needsUpdate = true;
  }
}
