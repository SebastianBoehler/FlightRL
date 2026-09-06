import * as T from "three/webgpu";

/** Release resources owned by a replaced scene, including shared mesh assets. */
export function disposeScene(root: T.Object3D) {
  const geometries = new Set<T.BufferGeometry>();
  const materials = new Set<T.Material>();
  const textures = new Set<T.Texture>();
  root.traverse((object) => {
    if (object instanceof T.Mesh || object instanceof T.Line || object instanceof T.Points || object instanceof T.Sprite) {
      geometries.add(object.geometry);
      for (const material of Array.isArray(object.material) ? object.material : [object.material]) {
        materials.add(material);
        for (const value of Object.values(material))
          if (value instanceof T.Texture) textures.add(value);
      }
    }
    if (object instanceof T.InstancedMesh) object.dispose();
    if (object instanceof T.DirectionalLight || object instanceof T.SpotLight)
      object.shadow.dispose();
  });
  textures.forEach((texture) => texture.dispose());
  materials.forEach((material) => material.dispose());
  geometries.forEach((geometry) => geometry.dispose());
}
