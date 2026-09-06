import * as T from "three/webgpu";
import { scannedMaps } from "./scanned-materials";

/** Dense local ground, then coarse rolling terrain continuing into atmospheric haze. */
export function forestTerrain() {
  const radii = [
    0.02,
    ...Array.from({ length: 50 }, (_, i) => (i + 1) * 0.7),
    40,
    48,
    60,
    80,
    110,
    160,
    240,
    360,
    550,
    800,
    1200,
  ];
  const positions: number[] = [],
    uv: number[] = [],
    indices: number[] = [],
    segments = 160;
  for (const r of radii)
    for (let j = 0; j <= segments; j++) {
      const angle = (j / segments) * Math.PI * 2,
        x = Math.cos(angle) * r,
        y = Math.sin(angle) * r;
      const blend = T.MathUtils.smoothstep(r, 30, 95);
      const z =
        -0.04 +
        0.035 * Math.sin(x * 0.7) * Math.sin(y * 0.8) +
        blend * (3 + 5 * Math.sin(x * 0.027) * Math.cos(y * 0.032));
      positions.push(x, y, z);
      uv.push(x * 0.5, y * 0.5);
    }
  for (let ring = 0; ring < radii.length - 1; ring++)
    for (let j = 0; j < segments; j++) {
      const a = ring * (segments + 1) + j,
        b = a + segments + 1;
      indices.push(a, b, a + 1, a + 1, b, b + 1);
    }
  // Fill the tiny central disk so every downward support query sees solid soil.
  const center = positions.length / 3;
  positions.push(0, 0, -0.04);
  uv.push(0, 0);
  for (let j = 0; j < segments; j++) indices.push(center, j, j + 1);
  const geometry = new T.BufferGeometry();
  geometry.setAttribute("position", new T.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("uv", new T.Float32BufferAttribute(uv, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  const mesh = new T.Mesh(
    geometry,
    new T.MeshStandardMaterial({
      ...scannedMaps("forrest_ground_03"),
      normalScale: new T.Vector2(0.6, 0.6),
      metalness: 1,
      roughness: 1,
    }),
  );
  mesh.receiveShadow = true;
  mesh.userData.contact = "solid";
  return mesh;
}
