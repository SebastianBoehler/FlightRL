import * as T from "three/webgpu";
import { random } from "./textures";
import { barkMaterial } from "./scanned-materials";

/** Visual habitat detail; recorded training collisions retain their original bounds. */
export function forestUnderstory(root: T.Group) {
  const rng = random(7182);
  const bark = barkMaterial();
  const wood = new T.MeshStandardMaterial({ color: 0xbca77c, roughness: 1 });
  const moss = new T.MeshStandardMaterial({ color: 0x536337, roughness: 1 });
  const ringMaterial = new T.MeshStandardMaterial({ color: 0x776044, roughness: 1, side: T.DoubleSide });
  const z = (x: number, y: number) => -.04 + .035 * Math.sin(x * .7) * Math.sin(y * .8);
  function branch(a: T.Vector3, b: T.Vector3, radius: number, tip: number, material: T.Material) {
    const mesh = new T.Mesh(new T.CylinderGeometry(tip, radius, a.distanceTo(b), 12), material);
    mesh.position.copy(a).add(b).multiplyScalar(.5);
    mesh.quaternion.setFromUnitVectors(new T.Vector3(0, 1, 0), b.clone().sub(a).normalize());
    mesh.userData.contact = "solid";
    mesh.castShadow = mesh.receiveShadow = true;
    root.add(mesh);
  }
  // Distinct deadwood forms with exposed end grain, broken limbs and moss.
  for (let i = 0; i < 36; i++) {
    const x = -12 + rng() * 27;
    let y = -12 + rng() * 24;
    if (Math.abs(y + 1.5) < 1.1) y += 2.4;
    const radius = .14 + rng() * .2;
    const base = new T.Vector3(x, y, z(x, y));
    if (i % 3 === 0) {
      const height = .22 + rng() * .5;
      branch(base, base.clone().add(new T.Vector3(0, 0, height)), radius, radius * .87, bark);
      const cap = new T.Mesh(new T.CircleGeometry(radius * .86, 24), wood);
      cap.position.copy(base).add(new T.Vector3(0, 0, height + .002));
      cap.receiveShadow = true;
      root.add(cap);
      for (let j = 1; j < 7; j++) {
        const r = radius * .12 * j;
        const ring = new T.Mesh(new T.RingGeometry(r, r + .004, 32), ringMaterial);
        ring.position.copy(cap.position).add(new T.Vector3(0, 0, .001));
        root.add(ring);
      }
    } else {
      const angle = rng() * Math.PI * 2;
      const direction = new T.Vector3(Math.cos(angle), Math.sin(angle), 0);
      const start = base.clone().add(new T.Vector3(0, 0, radius));
      const end = start.clone().addScaledVector(direction, 1.6 + rng() * 2.5);
      end.z = z(end.x, end.y) + radius * .7;
      branch(start, end, radius, radius * .7, bark);
      const cap = new T.Mesh(new T.CircleGeometry(radius * .94, 24), wood);
      cap.position.copy(start).addScaledVector(direction, -.002);
      cap.quaternion.setFromUnitVectors(new T.Vector3(0, 0, 1), direction.clone().negate());
      root.add(cap);
      for (let j = 0; j < 4; j++) {
        const a = start.clone().lerp(end, .15 + rng() * .7);
        const b = a.clone().add(new T.Vector3((rng()-.5)*.7, (rng()-.5)*.7, .18 + rng()*.35));
        branch(a, b, .035, .009, bark);
        const patch = new T.Mesh(new T.SphereGeometry(1, 8, 6), moss);
        patch.position.copy(a).add(new T.Vector3(0, 0, radius * .8));
        patch.scale.set(radius * 1.4, radius * .7, .045);
        root.add(patch);
      }
    }
  }
  // Fern rosettes: many paired tapered leaflets along curved fronds.
  const vertices: number[] = [];
  for (let j = 1; j <= 10; j++) {
    const t = j / 11, reach = .12 * Math.sin(t * Math.PI);
    for (const side of [-1, 1]) {
      const y = t * .65, h = Math.sin(t * Math.PI * .8) * .22;
      vertices.push(0, y-.035, h, side*reach, y+.035, h+.018, 0, y+.045, h+.02);
    }
  }
  const geometry = new T.BufferGeometry();
  geometry.setAttribute("position", new T.Float32BufferAttribute(vertices, 3));
  geometry.computeVertexNormals();
  const ferns = new T.InstancedMesh(geometry, new T.MeshStandardMaterial({ color: 0x527638, roughness: .9, side: T.DoubleSide }), 2400);
  const transform = new T.Object3D();
  for (let i = 0; i < 300; i++) {
    const x = -13 + rng()*29, y = -13 + rng()*26;
    const scale = Math.abs(y+1.5) < 1 ? .2 : .6+rng();
    for (let j = 0; j < 8; j++) {
      transform.position.set(x, y, z(x,y)+.015);
      transform.rotation.set(0, 0, j*Math.PI/4 + i);
      transform.scale.setScalar(scale*(.7+rng()*.3));
      transform.updateMatrix();
      ferns.setMatrixAt(i*8+j, transform.matrix);
      ferns.setColorAt(i*8+j, new T.Color().setHSL(.23+rng()*.08, .35, .25+rng()*.12));
    }
  }
  ferns.castShadow = ferns.receiveShadow = true;
  root.add(ferns);
}
