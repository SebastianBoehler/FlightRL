import * as T from "three/webgpu";

const images = new Map<string, HTMLImageElement>();
const assets = ["bark_willow", "forrest_ground_03"];
export async function prepareForestTextures() {
  await Promise.all(
    assets.flatMap((asset) =>
      ["diff", "nor_gl", "arm"].map(async (kind) => {
        const name = `${asset}_${kind}_1k.jpg`;
        if (!images.has(name))
          images.set(
            name,
            await new T.ImageLoader().loadAsync(`/assets/forest/${name}`),
          );
      }),
    ),
  );
}
export function scannedMaps(asset: string, repeat = 1) {
  const maps = ["diff", "nor_gl", "arm"].map((kind) => {
    const image = images.get(`${asset}_${kind}_1k.jpg`);
    if (!image)
      throw Error(
        "Scanned forest textures must finish loading before scene creation",
      );
    const texture = new T.Texture(image);
    texture.colorSpace = kind === "diff" ? T.SRGBColorSpace : T.NoColorSpace;
    texture.wrapS = texture.wrapT = T.RepeatWrapping;
    texture.repeat.setScalar(repeat);
    texture.anisotropy = 4;
    texture.needsUpdate = true;
    return texture;
  });
  return {
    map: maps[0],
    normalMap: maps[1],
    aoMap: maps[2],
    roughnessMap: maps[2],
    metalnessMap: maps[2],
  };
}
export function barkMaterial() {
  return new T.MeshStandardMaterial({
    ...scannedMaps("bark_willow"),
    roughness: 1,
    metalness: 1,
    normalScale: new T.Vector2(0.6, 0.6),
  });
}
