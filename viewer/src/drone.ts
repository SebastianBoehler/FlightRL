import * as THREE from "three/webgpu";
import type { World } from "./world";
const material = (color: number, metalness = 0) =>
  new THREE.MeshStandardMaterial({ color, metalness });
export function buildDrone(world: World) {
  world.box(
    [0.09, 0.06, 0.025],
    [0, 0, 0],
    material(0xd7e0e2, 0.65),
    world.drone,
  );
  for (const x of [-0.049, 0.049])
    for (const y of [-0.049, 0.049]) {
      const arm = world.box(
        [0.09, 0.009, 0.009],
        [x / 2, y / 2, 0],
        material(0x18272d, 0.8),
        world.drone,
      );
      arm.rotation.z = Math.atan2(y, x);
      const rotor = new THREE.Mesh(
        new THREE.CylinderGeometry(0.026, 0.026, 0.004, 24),
        new THREE.MeshStandardMaterial({
          color: 0x79decc,
          transparent: true,
          opacity: 0.55,
        }),
      );
      rotor.rotation.x = Math.PI / 2;
      rotor.position.set(x, y, 0.008);
      world.drone.add(rotor);
    }
  const halo = new THREE.Mesh(
    new THREE.RingGeometry(0.15, 0.16, 48),
    new THREE.MeshBasicMaterial({ color: 0x59d7c5, side: THREE.DoubleSide }),
  );
  halo.position.z = -0.04;
  world.drone.add(halo);
}
