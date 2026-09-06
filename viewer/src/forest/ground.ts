import * as T from "three/webgpu";
import { random } from "./textures";
import { forestTerrain } from "./terrain";
import { forestCabin } from "./cabin";
import { forestUnderstory } from "./understory";
export function forestGround(root: T.Group) {
  forestUnderstory(root);
  forestCabin(root);
  const rng = random(614);
  root.add(forestTerrain());
  const blade = new T.BufferGeometry();
  blade.setAttribute(
    "position",
    new T.Float32BufferAttribute(
      [-0.012, 0, 0, 0.012, 0, 0, 0.004, 0.018, 0.16, -0.008, 0.025, 0.08],
      3,
    ),
  );
  blade.setIndex([0, 1, 2, 0, 2, 3]);
  blade.computeVertexNormals();
  const grass = new T.InstancedMesh(
    blade,
    new T.MeshStandardMaterial({
      color: 0x596734,
      roughness: 1,
      side: T.DoubleSide,
    }),
    18000,
  );
  const o = new T.Object3D();
  for (let i = 0; i < 18000; i++) {
    const x = -18 + rng() * 40,
      y = -20 + rng() * 40;
    const path = Math.exp(
      -Math.pow((y + 1.5 + Math.sin(x * 0.2) * 0.25) / 0.6, 2),
    );
    o.position.set(x, y, -0.018);
    o.rotation.z = rng() * 6.28;
    const meadow = Math.max(0, Math.sin(x * .38) * Math.cos(y * .32));
    o.scale.set(1 + rng(), 1 + rng(), (0.5 + rng() * 1.8 + meadow * 4) * (1 - path * 0.96));
    o.updateMatrix();
    grass.setMatrixAt(i, o.matrix);
    grass.setColorAt(
      i,
      new T.Color().setHSL(0.19 + rng() * 0.08, 0.3, 0.2 + rng() * 0.15),
    );
  }
  grass.receiveShadow = true;
  root.add(grass);
  const stone = new T.IcosahedronGeometry(1, 1);
  const rocks = new T.InstancedMesh(
    stone,
    new T.MeshStandardMaterial({
      color: 0x727368,
      roughness: 1,
      flatShading: true,
    }),
    400,
  );
  for (let i = 0; i < 400; i++) {
    o.position.set(-15 + rng() * 35, -16 + rng() * 32, 0.015);
    o.rotation.set(rng() * 3, rng() * 3, rng() * 3);
    o.scale.set(0.03 + rng() * 0.1, 0.03 + rng() * 0.08, 0.02 + rng() * 0.06);
    o.updateMatrix();
    rocks.setMatrixAt(i, o.matrix);
  }
  rocks.userData.contact = "solid";
  rocks.castShadow = rocks.receiveShadow = true;
  root.add(rocks);
}
