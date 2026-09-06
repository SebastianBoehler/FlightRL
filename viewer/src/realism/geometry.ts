import * as T from "three/webgpu";
import type { SharedScene, RigidBodySpec } from "./types";

function encode(array: Float32Array | Uint32Array) {
  const bytes = new Uint8Array(array.buffer),
    parts: string[] = [];
  for (let i = 0; i < bytes.length; i += 8192)
    parts.push(String.fromCharCode(...bytes.subarray(i, i + 8192)));
  return btoa(parts.join(""));
}
/** Export the actual solid triangles after all authoring transforms. Foliage is soft. */
export function sharedScene(
  root: T.Object3D,
  bodies: RigidBodySpec[],
  materialAssets: unknown,
  wind: number[],
): SharedScene {
  root.updateMatrixWorld(true);
  const vertices: number[] = [],
    indices: number[] = [];
  const matrix = new T.Matrix4(),
    instance = new T.Matrix4(),
    p = new T.Vector3();
  root.traverse((object) => {
    if (!(object instanceof T.Mesh) || object.userData.contact !== "solid")
      return;
    const geometry = object.geometry,
      attribute = geometry.attributes.position;
    const count = object instanceof T.InstancedMesh ? object.count : 1;
    for (let j = 0; j < count; j++) {
      matrix.copy(object.matrixWorld);
      if (object instanceof T.InstancedMesh) {
        object.getMatrixAt(j, instance);
        matrix.multiply(instance);
      }
      const offset = vertices.length / 3;
      for (let k = 0; k < attribute.count; k++) {
        p.fromBufferAttribute(attribute, k).applyMatrix4(matrix);
        vertices.push(p.x, p.y, p.z);
      }
      const n = geometry.index?.count ?? attribute.count;
      for (let k = 0; k < n; k += 3) {
        const ids = [0, 1, 2].map(
          (d) => offset + (geometry.index?.getX(k + d) ?? k + d),
        );
        const [a, b, c] = ids.map((i) =>
          new T.Vector3().fromArray(vertices, i * 3),
        );
        if (b.sub(a).cross(c.sub(a)).lengthSq() > 1e-12) indices.push(...ids);
      }
    }
  });
  if (!indices.length)
    throw Error("Shared forest has no solid collision triangles");
  return {
    schema: "flightrl.shared_forest.v1",
    units: "m",
    up: "z",
    quaternionOrder: "xyzw",
    wind_m_s: wind,
    vertices: encode(new Float32Array(vertices)),
    indices: encode(new Uint32Array(indices)),
    triangleCount: indices.length / 3,
    bodies,
    materialAssets,
  };
}
