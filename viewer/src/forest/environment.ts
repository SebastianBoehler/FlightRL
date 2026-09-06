import * as T from "three/webgpu";
import { SkyMesh } from "three/addons/objects/SkyMesh.js";
import { HDRLoader } from "three/addons/loaders/HDRLoader.js";

export async function forestEnvironment(
  renderer: T.WebGPURenderer,
  scene: T.Scene,
) {
  const hdr = await new HDRLoader().loadAsync(
    "/assets/forest/spruit_sunrise_1k.hdr",
  );
  hdr.mapping = T.EquirectangularReflectionMapping;
  const generator = new T.PMREMGenerator(renderer);
  // The installed Three.js runtime exposes this method; its matching type package omits it.
  const environment = (
    generator as T.PMREMGenerator & {
      fromEquirectangular(texture: T.Texture): T.RenderTarget;
    }
  ).fromEquirectangular(hdr);
  scene.environment = environment.texture;
  scene.environmentIntensity = 0.7;
  scene.environmentRotation.set(Math.PI / 2, 0, -0.5);
  scene.background = null;
  const sky = new SkyMesh();
  sky.scale.setScalar(5000);
  sky.upUniform.value.set(0, 0, 1);
  sky.sunPosition.value.set(-12, -9, 6);
  sky.turbidity.value = 5;
  sky.rayleigh.value = 1.5;
  sky.material.fog = false;
  sky.frustumCulled = false;
  sky.renderOrder = -100;
  scene.add(sky);
  scene.fog = new T.FogExp2(0xc4c9be, 0.018);
  await renderer.waitForGPU();
  generator.dispose();
  return () => {
    scene.remove(sky);
    sky.geometry.dispose();
    sky.material.dispose();
    hdr.dispose();
    environment.dispose();
  };
}
