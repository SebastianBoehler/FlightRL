import * as T from "three/webgpu";

export function sunrise(scene: T.Scene) {
  scene.background = new T.Color(0xd9bca0);
  scene.fog = new T.FogExp2(0xc8b09a, .018);
  scene.add(new T.HemisphereLight(0xa9c3dc, 0x514b35, 1.8));
  const sun = new T.DirectionalLight(0xffb46b, 4.5);
  sun.position.set(-12, -9, 6);
  sun.target.position.set(1,0,0);
  sun.castShadow = true;
  sun.shadow.mapSize.set(4096,4096);
  Object.assign(sun.shadow.camera, {left:-22,right:22,top:22,bottom:-22,near:.5,far:65});
  sun.shadow.bias=-.00008;
  sun.shadow.normalBias=.025;
  scene.add(sun,sun.target);
}
