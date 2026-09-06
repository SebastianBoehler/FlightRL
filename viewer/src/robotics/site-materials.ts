import * as T from "three/webgpu";

export function concreteTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 256;
  const ctx = canvas.getContext("2d")!,
    image = ctx.createImageData(256, 256);
  for (let y = 0; y < 256; y++)
    for (let x = 0; x < 256; x++) {
      let hash = Math.imul(x + y * 256 + 1, 0x45d9f3b);
      hash = Math.imul(hash ^ (hash >>> 16), 0x45d9f3b);
      const grain = ((hash ^ (hash >>> 16)) >>> 0) / 0xffffffff - 0.5;
      const seam = x < 2 || y < 2;
      const v = seam ? 99 : 147 + grain * 18;
      image.data.set([v, v + 2, v, 255], (y * 256 + x) * 4);
    }
  ctx.putImageData(image, 0, 0);
  const map = new T.CanvasTexture(canvas);
  map.colorSpace = T.SRGBColorSpace;
  map.wrapS = map.wrapT = T.RepeatWrapping;
  map.repeat.set(60, 50);
  map.anisotropy = 4;
  return map;
}

export function siteMaterial(
  name: string,
  color: number[],
  concrete: T.Texture,
) {
  const floor = name === "floor",
    rubber = name.includes("tire"),
    signal = name.startsWith("signal_");
  return new T.MeshStandardMaterial({
    color: new T.Color().fromArray(color),
    roughness: floor ? 0.9 : rubber ? 0.97 : 0.43,
    metalness: floor || rubber ? 0 : 0.5,
    map: floor ? concrete : null,
    emissive: signal ? new T.Color().fromArray(color) : 0,
    emissiveIntensity: signal ? 0.35 : 0,
  });
}
