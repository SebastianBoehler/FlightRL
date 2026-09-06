import * as T from "three/webgpu";

interface Part {
  name: string; color: number[]; roughness: number; metalness: number;
  position: number[]; vertices: number[]; indices: number[];
}
export interface DroneReference {
  id: "fpv" | "agriculture";
  name: string; mass_kg: number; dimensions_m: number[];
  rotor_radius_m: number; rotor_centers_m: number[][];
  camera_offset_m: number[]; source: string; provenance: string;
  payload: string; assumptions: string;
}
export interface DroneAsset extends DroneReference { parts: Part[] }

export async function loadDrone(kind: DroneReference["id"]): Promise<DroneAsset> {
  const response = await fetch(`/assets/drone-models/${kind}.json`);
  if (!response.ok) throw Error(`Drone reference unavailable: ${kind}`);
  const asset: DroneAsset = await response.json();
  if (asset.id !== kind || !asset.parts.length) throw Error("Invalid drone reference asset");
  return asset;
}

export function droneMaterial(name: string, color: number[]) {
  return new T.MeshStandardMaterial({
    color: new T.Color().fromArray(color),
    roughness: name.endsWith("rubber") ? .86 : name.endsWith("glass") ? .12 : .5,
    metalness: name.endsWith("metal") ? .75 : name.endsWith("glass") ? .45 : .08,
  });
}

/** Visual blade phase uses acquisition time, never wall time or invented RPM telemetry. */
export function animateRotors(rotors: T.Object3D[], time: number) {
  rotors.forEach((rotor, i) => {
    rotor.rotation.z = time * 37 * (i === 0 || i === 3 ? 1 : -1);
  });
}

export function droneGeometry(asset: DroneAsset) {
  const body = new T.Group(), rotors: T.Object3D[] = [];
  for (const part of asset.parts) {
    const geometry = new T.BufferGeometry();
    geometry.setAttribute("position", new T.Float32BufferAttribute(part.vertices, 3));
    geometry.setIndex(part.indices); geometry.computeVertexNormals();
    const material = new T.MeshStandardMaterial({
      color: new T.Color().fromArray(part.color), roughness: part.roughness,
      metalness: part.metalness,
    });
    const mesh = new T.Mesh(geometry, material);
    mesh.position.fromArray(part.position);
    mesh.castShadow = mesh.receiveShadow = true;
    body.add(mesh);
    if (part.name.startsWith("rotor_")) rotors.push(mesh);
  }
  return {body, update: (time: number) => animateRotors(rotors, time)};
}
