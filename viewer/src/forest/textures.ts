import * as T from "three/webgpu";
export function random(seed = 71) {
  return () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
}
function canvas(size: number) {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  return c;
}
function texture(c: HTMLCanvasElement, repeat = 1) {
  const t = new T.CanvasTexture(c);
  t.colorSpace = T.SRGBColorSpace;
  t.wrapS = t.wrapT = T.RepeatWrapping;
  t.repeat.set(repeat, repeat);
  t.anisotropy = 8;
  return t;
}
export function forestTextures() {
  const leaf = canvas(128),
    l = leaf.getContext("2d")!;
  const grad = l.createLinearGradient(20, 0, 100, 120);
  grad.addColorStop(0, "#718d31");
  grad.addColorStop(0.5, "#365b1d");
  grad.addColorStop(1, "#96a847");
  l.fillStyle = grad;
  l.beginPath();
  l.moveTo(64, 4);
  l.bezierCurveTo(119, 43, 105, 93, 64, 122);
  l.bezierCurveTo(15, 92, 8, 47, 64, 4);
  l.fill();
  l.strokeStyle = "rgba(182,192,102,.5)";
  l.lineWidth = 1.4;
  l.beginPath();
  l.moveTo(64, 8);
  l.lineTo(64, 119);
  l.stroke();
  for (let y = 30; y < 110; y += 15) {
    l.beginPath();
    l.moveTo(64, y + 12);
    l.lineTo(31, y - 9);
    l.moveTo(64, y + 12);
    l.lineTo(97, y - 9);
    l.stroke();
  }
  return {
    leaf: texture(leaf),
  };
}
