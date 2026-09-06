import * as T from "three/webgpu";
import { buildForest } from "../forest/trees";
import { forestGround } from "../forest/ground";
import { sunrise } from "../forest/sunrise";
import {
  prepareForestTextures,
  barkMaterial,
} from "../forest/scanned-materials";
import { scannedStumps } from "../forest/scanned-models";
import { forestEnvironment } from "../forest/environment";
import { sharedScene } from "./geometry";
import { FlightGuides } from "./flight-guides";
import { ParticleView } from "./particles";
import { loadDrone, droneGeometry } from "../models/drone";
import type { Episode } from "../types";
import type { RigidBodySpec, WorldState } from "./types";

export async function createForest(renderer: T.WebGPURenderer) {
  await prepareForestTextures();
  const response = await fetch("/data/forest-held-out.json");
  if (!response.ok) throw Error("Forest source episode unavailable");
  const episode: Episode = await response.json();
  const wind = episode.scene.environment?.wind_m_s;
  if (!wind) throw Error("Forest wind metadata missing");
  const drone = await loadDrone("agriculture");
  const spawns = [-4, 0, 4].map(y => [-4, y, 2.3]);
  const scene = new T.Scene(),
    forest = buildForest(wind, spawns.map(([x, y]) => ({x, y, radius: 4.2})));
  scene.add(forest.root);
  forestGround(forest.root);
  await scannedStumps(forest.root);
  sunrise(scene);
  forest.loose.visible = false; // Live leaf positions come from the native particle simulation.
  const disposeEnvironment = await forestEnvironment(renderer, scene);
  const colors = [0xd05c43, 0x3e8bc3, 0x6faa62];
  const specs: RigidBodySpec[] = [
    ...spawns.map((position, i) => ({
      id: `drone-${i}`,
      position,
      vehicle: drone.id,
      model: {...drone, parts: undefined},
      quaternion: [0, 0, 0, 1],
      halfExtents: drone.dimensions_m.map(x => x / 2),
      mass: drone.mass_kg,
    })),
    ...[0, 1, 2].map((i) => ({
      id: `debris-${i}`,
      position: [0.5 + i * 0.6, -2.6, 1.4 + i * 0.3],
      quaternion: [0, 0, 0, 1],
      halfExtents: [0.16, 0.12, 0.1],
      mass: 0.8,
    })),
  ];
  const vehicles = spawns.map(() => droneGeometry(drone));
  const bodies = specs.map((spec, i) => {
    const body = i < 3 ? vehicles[i].body : new T.Group();
    body.position.fromArray(spec.position);
    body.quaternion.fromArray(spec.quaternion);
    if (i >= 3) {
      const mesh = new T.Mesh(new T.BoxGeometry(
        ...(spec.halfExtents.map(x => x * 2) as [number, number, number]),
      ), barkMaterial());
      mesh.castShadow = mesh.receiveShadow = true;
      body.add(mesh);
    }
    scene.add(body);
    return body;
  });
  for (let i = 0; i < 3; i++) {
    const beacon = new T.Mesh(
      new T.BoxGeometry(0.035, 0.55, 0.55),
      new T.MeshStandardMaterial({ color: colors[i], roughness: 0.75 }),
    );
    beacon.position.set(5, (i - 1) * 1.5, 1.6);
    beacon.userData.contact = "solid";
    beacon.castShadow = true;
    beacon.receiveShadow = true;
    forest.root.add(beacon);
  }
  const manifest = await fetch("/assets/forest/manifest.json");
  if (!manifest.ok) throw Error("Forest material provenance missing");
  const description = sharedScene(
    forest.root,
    specs,
    await manifest.json(),
    wind,
  );
  const particles = new ParticleView();
  scene.add(particles);
  const guides = new FlightGuides();
  scene.add(guides);
  const initial: WorldState = {
    wind_m_s: wind,
    sequence: 0,
    time_s: 0,
    positions: specs.map((x) => x.position),
    quaternions: specs.map((x) => x.quaternion),
    velocities: [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ],
    rates: [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ],
    mode: "preparing",
    contacts: 0,
  };
  const shadows: T.LightShadow[] = [];
  scene.traverse((o) => {
    if (o instanceof T.DirectionalLight && o.castShadow) {
      o.shadow.autoUpdate = false;
      shadows.push(o.shadow);
    }
  });
  function apply(state: WorldState) {
    guides.update(state);
    vehicles.forEach(v => v.update(state.time_s));
    bodies.forEach((body, i) => {
      body.position.fromArray(state.positions[i]);
      body.quaternion.fromArray(state.quaternions[i]);
    });
    forest.update(state.time_s, state.wind_m_s);
    if (state.particles && state.particleKinds)
      particles.update(state.particles, state.particleKinds, state.time_s);
    shadows.forEach((s) => (s.needsUpdate = true));
  }
  return {
    scene,
    drone,
    guides,
    forest,
    bodies,
    description,
    initial,
    apply,
    disposeEnvironment,
  };
}
