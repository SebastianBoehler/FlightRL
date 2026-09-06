import * as T from "three/webgpu";
import { forestEnvironment } from "../forest/environment";
import { sunrise } from "../forest/sunrise";
import { animateRotors, droneMaterial, type DroneReference } from "../models/drone";
import { concreteTexture, siteMaterial } from "./site-materials";

interface GeometrySpec {
  name: string;
  body: number;
  type: string;
  size: number[];
  position: number[];
  quaternion: number[];
  color: number[];
  mesh?: number;
}
export interface SceneDescription {
  meshes: Record<number, { vertices: number[]; indices: number[] }>;
  cameras: Array<{ id: string; body: number; robot_id: string }>;
  body_count: number;
  drone_reference: DroneReference;
  geometries: GeometrySpec[];
  targets: Array<{ id: number; asset?: string }>;
  site?: {
    name: string;
    seed: number;
    variant: number;
    sun_angle: number;
    haze: number;
  } | null;
}
function primitive(type: string, spec: GeometrySpec, meshes: SceneDescription["meshes"]) {
  if (type === "mesh") {
    const source = meshes[spec.mesh!];
    if (!source) throw Error(`Missing compiled mesh ${spec.mesh}`);
    const geometry = new T.BufferGeometry();
    geometry.setAttribute("position", new T.Float32BufferAttribute(source.vertices, 3));
    geometry.setIndex(source.indices); geometry.computeVertexNormals();
    return geometry;
  }
  if (type === "capsule") {
    const geometry = new T.CapsuleGeometry(spec.size[0], spec.size[1] * 2, 8, 24);
    geometry.rotateX(Math.PI / 2); return geometry;
  }
  if (type === "box") return new T.BoxGeometry(2, 2, 2);
  if (type === "sphere") return new T.SphereGeometry(1, 16, 10);
  if (type !== "cylinder") throw Error(`Unsupported physical shape: ${type}`);
  const geometry = new T.CylinderGeometry(1, 1, 2, 96);
  geometry.rotateX(Math.PI / 2);
  return geometry;
}
function scale(spec: GeometrySpec) {
  if (spec.type === "mesh" || spec.type === "capsule") return [1, 1, 1];
  return spec.type === "box"
    ? spec.size
    : spec.type === "sphere"
      ? [spec.size[0], spec.size[0], spec.size[0]]
      : [spec.size[0], spec.size[0], spec.size[1]];
}
export async function industrialScene(
  renderer: T.WebGPURenderer,
  description: SceneDescription,
) {
  const scene = new T.Scene(),
    bodies = Array.from(
      { length: description.body_count },
      () => new T.Group(),
    );
  bodies.forEach((b) => scene.add(b));
  sunrise(scene);
  const disposeEnvironment = await forestEnvironment(renderer, scene);
  if (description.site) {
    scene.fog = new T.FogExp2(0xb5c1c3, description.site.haze);
    scene.environmentIntensity = 0.55;
    scene.traverse((object) => {
      if (object instanceof T.DirectionalLight) {
        object.intensity = 3.2;
        object.position.set(-12, -8 + description.site!.sun_angle * 15, 12);
      }
    });
  }
  const tags = new Map<number, T.Texture>(),
    loader = new T.TextureLoader();
  await Promise.all(
    description.targets.map(async (t) => {
      const map = await loader.loadAsync(`/assets/markers/${t.id}.png`);
      map.colorSpace = T.SRGBColorSpace;
      tags.set(t.id, map);
    }),
  );
  const rotors: T.Object3D[] = [];
  const concrete = concreteTexture(),
    groups = new Map<string, GeometrySpec[]>();
  for (const spec of description.geometries) {
    if (spec.body === 0 && !spec.name.startsWith("marker_")) {
      const category =
        spec.name === "floor"
          ? "floor"
          : spec.name.startsWith("signal_")
            ? "signal"
            : "metal";
      const key = JSON.stringify([spec.type, spec.mesh, category, spec.color]);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)!.push(spec);
      continue;
    }
    let material: T.Material | T.Material[] = siteMaterial(
      spec.name,
      spec.color,
      concrete,
    );
    if (spec.name.startsWith("marker_")) {
      const id = Number(spec.name.split("_")[1]);
      const face = new T.MeshBasicMaterial({ map: tags.get(id) });
      material = [
        face,
        face,
        ...Array.from(
          { length: 4 },
          () => new T.MeshStandardMaterial({ color: 0xe8e6df }),
        ),
      ];
    }
    if (spec.name.startsWith("drone_visual_")) material = droneMaterial(spec.name, spec.color);
    const mesh = new T.Mesh(primitive(spec.type, spec, description.meshes), material);
    mesh.position.fromArray(spec.position);
    mesh.quaternion.fromArray(spec.quaternion);
    mesh.scale.fromArray(scale(spec));
    mesh.castShadow = mesh.receiveShadow = true;
    if (spec.name.startsWith("drone_visual_rotor_")) {
      const i = Number(spec.name.split("_").at(-1));
      const pivot = new T.Group();
      pivot.position.fromArray(description.drone_reference.rotor_centers_m[i]);
      mesh.position.sub(pivot.position);
      pivot.add(mesh); bodies[spec.body - 1].add(pivot); rotors[i] = pivot;
    } else (spec.body === 0 ? scene : bodies[spec.body - 1]).add(mesh);
  }
  const transform = new T.Object3D();
  for (const specs of groups.values()) {
    const first = specs[0],
      mesh = new T.InstancedMesh(
        primitive(first.type, first, description.meshes),
        siteMaterial(first.name, first.color, concrete),
        specs.length,
      );
    specs.forEach((spec, i) => {
      transform.position.fromArray(spec.position);
      transform.quaternion.fromArray(spec.quaternion);
      transform.scale.fromArray(scale(spec));
      transform.updateMatrix();
      mesh.setMatrixAt(i, transform.matrix);
    });
    mesh.castShadow = mesh.receiveShadow = true;
    mesh.computeBoundingSphere();
    scene.add(mesh);
  }
  function apply(state: { positions: number[][]; quaternions: number[][] }, time: number) {
    animateRotors(rotors, time);
    bodies.forEach((b, i) => {
      b.position.fromArray(state.positions[i]);
      b.quaternion.fromArray(state.quaternions[i]);
    });
  }
  return { scene, bodies, apply, disposeEnvironment };
}
