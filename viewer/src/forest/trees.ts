import * as T from "three/webgpu";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { positionLocal, vec3, sin, uniform, float } from "three/tsl";
import { random, forestTextures } from "./textures";
import { forestLayout, type Clearing } from "./layout";
import { barkMaterial } from "./scanned-materials";
export interface ForestAssets {
  root: T.Group;
  leaves: T.InstancedMesh;
  loose: T.InstancedMesh;
  update: (time: number, wind: number[]) => void;
}
export function buildForest(ambientWind: number[], clearings: Clearing[] = []): ForestAssets {
  const root = new T.Group(),
    rng = random(911),
    maps = forestTextures();
  const bark = barkMaterial();
  const foliage = new T.MeshSSSNodeMaterial({
    map: maps.leaf, alphaTest: .48, side: T.DoubleSide, roughness: .9, color: 0xffffff,
  });
  foliage.thicknessColorNode = vec3(.18, .27, .05);
  foliage.thicknessDistortionNode = float(.25);
  foliage.thicknessAmbientNode = float(.02);
  foliage.thicknessAttenuationNode = float(.45);
  foliage.thicknessPowerNode = float(3);
  foliage.thicknessScaleNode = float(.7);
  const phase = uniform(0);
  foliage.positionNode = positionLocal.add(
    vec3(sin(phase.add(positionLocal.y.mul(4))).mul(0.06), 0, 0),
  );
  const stems = forestLayout(clearings);
  const instances: number[][] = [];
  const wood: T.BufferGeometry[] = [];
  function branch(a: T.Vector3, b: T.Vector3, r: number) {
    const delta = b.clone().sub(a);
    const m = new T.Mesh(
      new T.CylinderGeometry(r * 0.45, r, delta.length(), 7),
      bark,
    );
    const uv = m.geometry.attributes.uv;
    for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * Math.PI * 2 * r, uv.getY(i) * delta.length());
    m.position.copy(a).add(b).multiplyScalar(0.5);
    m.quaternion.setFromUnitVectors(new T.Vector3(0, 1, 0), delta.normalize());
    m.updateMatrix();
    m.geometry.applyMatrix4(m.matrix);
    wood.push(m.geometry);
  }
  for (const tree of stems) {
    if (tree.y < -3 && Math.hypot(tree.x - 5, tree.y + 7) < 5) continue;
    const base = new T.Vector3(tree.x, tree.y, 0),
      top = new T.Vector3(tree.x + tree.leanX, tree.y + tree.leanY, tree.h);
    branch(base, top, tree.r);
    for (let j = 0; j < 5; j++) {
      const angle = j * 1.256 + rng() * 0.4;
      branch(
        new T.Vector3(
          tree.x + Math.cos(angle) * tree.r * 3,
          tree.y + Math.sin(angle) * tree.r * 3,
          0.015,
        ),
        new T.Vector3(tree.x, tree.y, 0.4),
        tree.r * 0.36,
      );
    }
    for (let j = 0; j < 14; j++) {
      const angle = j * 2.4 + rng() * 0.4,
        z = tree.h * (0.52 + (0.43 * j) / 14),
        length = (0.5 + rng() * 0.85) * (.65 + tree.h * .07);
      const start = new T.Vector3(tree.x + tree.leanX * z / tree.h, tree.y + tree.leanY * z / tree.h, z),
        end = new T.Vector3(
          start.x + Math.cos(angle) * length,
          start.y + Math.sin(angle) * length,
          z + 0.3 + rng() * 0.4,
        );
      branch(start, end, tree.r * 0.25);
      for (let twig = 0; twig < 3; twig++) {
        const a = angle + (twig - 1) * 0.6;
        const tip = end
          .clone()
          .add(new T.Vector3(Math.cos(a) * 0.4, Math.sin(a) * 0.4, 0.16));
        branch(end, tip, 0.014);
        for (let k = 0; k < 38; k++) {
          const az = rng() * 6.28,
            rr = Math.sqrt(rng()) * 0.55;
          instances.push([
            tip.x + Math.cos(az) * rr,
            tip.y + Math.sin(az) * rr,
            tip.z + (rng() - 0.5) * 0.6,
            rng() * 6.28,
            rng() * 0.8,
            0.09 + rng() * 0.1,
          ]);
        }
      }
    }
  }
  const woodMesh = new T.Mesh(mergeGeometries(wood), bark);
  woodMesh.userData.contact = "solid";
  woodMesh.castShadow = woodMesh.receiveShadow = true;
  root.add(woodMesh);
  wood.forEach((g) => g.dispose());
  const leaves = new T.InstancedMesh(
    new T.PlaneGeometry(1, 1.8),
    foliage,
    instances.length,
  );
  const obj = new T.Object3D();
  instances.forEach((p, i) => {
    obj.position.set(p[0], p[1], p[2]);
    obj.rotation.set(p[4], p[3], p[3] * 0.7);
    obj.scale.setScalar(p[5]);
    obj.updateMatrix();
    leaves.setMatrixAt(i, obj.matrix);
    leaves.setColorAt(
      i,
      new T.Color().setHSL(
        0.22 + rng() * 0.07,
        0.35 + rng() * 0.25,
        0.35 + rng() * 0.2,
      ),
    );
  });
  leaves.castShadow = leaves.receiveShadow = true;
  root.add(leaves);
  const loose = new T.InstancedMesh(
    new T.PlaneGeometry(0.055, 0.11),
    foliage,
    160,
  );
  root.add(loose);
  loose.castShadow = true;
  loose.instanceMatrix.setUsage(T.DynamicDrawUsage);
  loose.frustumCulled = false;
  const origins = Array.from({ length: 160 }, () => {
    const tree = stems[Math.floor(rng() * Math.min(14, stems.length))];
    return [
      tree.x + (rng() - 0.5),
      tree.y + (rng() - 0.5),
      tree.h * 0.7,
      rng() * 35,
      rng() * 6.28,
    ];
  });
  return {
    root,
    leaves,
    loose,
    update(time, wind) {
      phase.value = time * (0.8 + Math.hypot(...wind));
      // Finite leaves detach once, drag toward air velocity, flutter, then rest on soil.
      origins.forEach((p, i) => {
        const age = Math.max(0, time - p[3]);
        const z = Math.max(0.018, p[2] - 0.45 * age);
        const airborne = z > 0.018;
        obj.position.set(
          p[0] +
            ambientWind[0] * Math.min(age, p[2] / 0.45) +
            Math.sin(age * 2 + p[4]) * (airborne ? 0.12 : 0),
          p[1] + ambientWind[1] * Math.min(age, p[2] / 0.45),
          z,
        );
        obj.rotation.set(
          airborne ? age * 2 : 0,
          airborne ? Math.sin(age * 3) : 0,
          p[4] + (airborne ? age : 0),
        );
        obj.scale.setScalar(1);
        obj.updateMatrix();
        loose.setMatrixAt(i, obj.matrix);
      });
      loose.instanceMatrix.needsUpdate = true;
    },
  };
}
