import * as T from "three/webgpu";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

/** CC0 photogrammetry, with the same triangles exported for solid contacts. */
export async function scannedStumps(root: T.Group) {
  const asset = await new GLTFLoader().loadAsync(
    "/assets/forest/tree_stump_01/tree_stump_01_1k.gltf",
  );
  for (const [x, y, angle] of [
    [0, -3.5, 0.3],
    [3, -4, 2.1],
    [7, 2, 4.4],
  ]) {
    const placement = new T.Group(),
      stump = asset.scene.clone(true);
    stump.rotation.x = Math.PI / 2;
    placement.add(stump);
    placement.rotation.z = angle;
    const bounds = new T.Box3().setFromObject(placement);
    placement.position.set(x, y, -0.04 - bounds.min.z);
    placement.traverse((object) => {
      if (object instanceof T.Mesh) {
        object.castShadow = object.receiveShadow = true;
        object.userData.contact = "solid";
      }
    });
    root.add(placement);
  }
}
