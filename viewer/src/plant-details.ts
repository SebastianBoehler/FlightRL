import * as THREE from "three/webgpu";

// Observer decoration lies on conservative collision boxes; actor uses native surfaces.
export function plantDetails(root: THREE.Group, boxes: number[][]) {
  const steel = new THREE.MeshStandardMaterial({
    color: 0xb5bfc2,
    metalness: 0.8,
    roughness: 0.32,
  });
  const dark = new THREE.MeshStandardMaterial({
    color: 0x15242c,
    roughness: 0.5,
  });
  for (const b of boxes) {
    const sx = b[1] - b[0],
      sy = b[3] - b[2],
      sz = b[5] - b[4];
    const cx = (b[0] + b[1]) / 2,
      cy = (b[2] + b[3]) / 2;
    if (b[4] > 2) {
      const alongX = sx > sy,
        length = alongX ? sx : sy;
      for (let t = -length / 2 + 0.3; t < length / 2; t += 1.1) {
        const clamp = new THREE.Mesh(
          new THREE.BoxGeometry(
            alongX ? 0.035 : sx + 0.018,
            alongX ? sy + 0.018 : 0.035,
            sz + 0.018,
          ),
          steel,
        );
        clamp.position.set(
          cx + (alongX ? t : 0),
          cy + (alongX ? 0 : t),
          (b[4] + b[5]) / 2,
        );
        root.add(clamp);
      }
    } else if (sz < 2.9) {
      const plate = new THREE.Mesh(
        new THREE.BoxGeometry(0.012, sy * 0.55, 0.48),
        dark,
      );
      plate.position.set(b[0] - 0.01, cy, Math.min(sz - 0.35, 1.45));
      root.add(plate);
      const gauge = new THREE.Mesh(new THREE.CircleGeometry(0.1, 32), steel);
      gauge.rotation.y = -Math.PI / 2;
      gauge.position.copy(plate.position);
      gauge.position.x -= 0.008;
      root.add(gauge);
      const lamp = new THREE.Mesh(
        new THREE.SphereGeometry(0.026, 12, 8),
        new THREE.MeshStandardMaterial({
          color: 0x6de5b0,
          emissive: 0x2d956a,
          emissiveIntensity: 1,
        }),
      );
      lamp.position.set(b[0] - 0.03, cy + sy * 0.17, plate.position.z);
      root.add(lamp);
      for (const z of [0.08, sz - 0.05]) {
        const rim = new THREE.Mesh(
          new THREE.BoxGeometry(sx + 0.016, sy + 0.016, 0.03),
          steel,
        );
        rim.position.set(cx, cy, z);
        root.add(rim);
      }
    }
  }
}

export function setCutaway(
  root: THREE.Group,
  markers: THREE.Mesh[],
  active: boolean,
) {
  root.traverse((object) => {
    if (
      !(object instanceof THREE.Mesh) ||
      !(object.material instanceof THREE.MeshStandardMaterial)
    )
      return;
    const fade = active && !markers.includes(object);
    if (object.material.transparent !== fade)
      object.material.needsUpdate = true;
    object.material.transparent = fade;
    object.material.opacity = fade ? 0.16 : 1;
    object.material.depthWrite = !fade;
  });
}
