import * as THREE from "three/webgpu";

/** Observer primitives mirror analytic sensor shapes; collision uses their bounds. */
export function forestObject(parent: THREE.Group, b: number[]) {
  const trunk = b[4] < 0.01 && b[5] > 3;
  const radius = (b[1] - b[0]) / 2;
  const height = b[5] - b[4];
  const mesh = new THREE.Mesh(
    trunk
      ? new THREE.CylinderGeometry(radius, radius, height, 16)
      : new THREE.SphereGeometry(1, 16, 12),
    new THREE.MeshStandardMaterial({
      color: trunk ? 0x5b4128 : b[4] > 1 ? 0x306229 : 0x676c5b,
      roughness: 0.95,
    }),
  );
  if (trunk) mesh.rotation.x = Math.PI / 2;
  else mesh.scale.set(radius, (b[3] - b[2]) / 2, height / 2);
  mesh.position.set((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2);
  mesh.castShadow = mesh.receiveShadow = true;
  parent.add(mesh);
}
