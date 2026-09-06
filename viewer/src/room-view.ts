import * as THREE from "three/webgpu";
import type { World } from "./world";
const material = (color: number, metalness = 0, roughness = 0.65) =>
  new THREE.MeshStandardMaterial({ color, metalness, roughness });
export function roomView(world: World, room: number[], forest: boolean) {
  const sx = room[1] - room[0],
    sy = room[3] - room[2],
    cx = (room[0] + room[1]) / 2,
    cy = (room[2] + room[3]) / 2;
  world.box(
    [sx, sy, 0.12],
    [cx, cy, -0.06],
    material(forest ? 0x4a522f : 0x526069, 0.25),
  );
  if (!forest)
    world.box(
      [0.1, sy, 3],
      [room[1] + 0.05, cy, 1.5],
      material(0x879598, 0.35),
    );
  if (!forest)
    world.box([sx, 0.1, 3], [cx, room[3] + 0.05, 1.5], material(0x62757e, 0.3));
  // Two walls are cut away for the observer; sensor still sees all room walls.
  const grid = new THREE.GridHelper(20, 40, 0x8ba0a6, 0x61747a);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = 0.004;
  if (!forest) world.root.add(grid);
  for (let x = forest ? Infinity : room[0] + 0.2; x < room[1]; x += 1.2) {
    world.box([0.028, sy - 0.15, 0.006], [x, cy, 0.01], material(0x8d9a9d));
  }
}
