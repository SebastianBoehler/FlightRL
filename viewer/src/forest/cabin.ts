import * as T from "three/webgpu";
import { random } from "./textures";
import { barkMaterial } from "./scanned-materials";

/** Weathered clearing landmark, outside the recorded navigation plot. */
export function forestCabin(root: T.Group) {
  const group = new T.Group();
  group.position.set(5, -7, 0);
  root.add(group);
  const rng = random(338);
  const wood = barkMaterial();
  const roof = new T.MeshStandardMaterial({color: 0x64483a, metalness: .35, roughness: .9});
  const dark = new T.MeshStandardMaterial({color: 0x171c19, roughness: 1});
  const stone = new T.MeshStandardMaterial({color: 0x73736a, roughness: 1});
  function box(w: number, d: number, h: number, x: number, y: number, z: number, mat: T.Material) {
    const mesh = new T.Mesh(new T.BoxGeometry(w,d,h),mat);
    mesh.userData.contact = "solid";
    mesh.position.set(x,y,z); mesh.castShadow = mesh.receiveShadow = true;
    group.add(mesh); return mesh;
  }
  box(3.9,3.2,.18,0,0,.02,stone);
  box(3.5,2.8,.1,0,0,.12,dark);
  for(let i=0;i<21;i++) {
    const x=-1.8+i*.18;
    // Open doorway and a boarded window on the front elevation.
    if(x>-.45 && x<.45) box(.165,.12,.42,x,-1.48,2.1,wood);
    else if(x>.7 && x<1.4) {
      box(.165,.12,.8,x,-1.48,.5,wood);
      box(.165,.12,.65,x,-1.48,1.95,wood);
    } else box(.165,.12,2.2-rng()*.08,x,-1.48,1.18,wood);
    box(.165,.12,2.2,x,1.48,1.18,wood);
  }
  for(let i=0;i<17;i++) for(const side of [-1,1])
    box(.12,.165,2.2,side*1.88,-1.44+i*.18,1.18,wood);
  for(const side of [-1,1]) {
    const panel=box(2.25,3.55,.1,side*1.02,0,2.65,roof);
    panel.rotation.y=side*.43;
    for(let j=0;j<18;j++) {
      const seam=box(2.25,.025,.035,side*1.02,-1.7+j*.2,2.71,roof);
      seam.rotation.y=side*.43;
    }
  }
  const board=box(.95,.075,.13,1.1,-1.58,1.35,wood);board.rotation.y=.22;
  box(.45,.45,1.25,-1, .6,2.9,stone);
  // Leaning door and decayed porch make silhouettes less uniform.
  const door=box(.75,.09,1.7,-.48,-1.7,.88,wood);door.rotation.y=-.16;door.rotation.z=.2;
  for(let i=0;i<10;i++) box(.2,1,.09,-1+i*.22,-2,.16+rng()*.035,wood);
}
