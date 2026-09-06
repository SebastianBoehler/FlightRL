import { random } from "./textures";

export interface ForestStem {
  x: number;
  y: number;
  r: number;
  h: number;
  leanX: number;
  leanY: number;
}

/** One irregular stand across the whole forest, with glades and mixed ages. */
export interface Clearing { x: number; y: number; radius: number }
export function forestLayout(clearings: Clearing[] = []): ForestStem[] {
  const rng = random(4619),
    stems: ForestStem[] = [];
  for (let attempt = 0; stems.length < 108 && attempt < 8000; attempt++) {
    const angle = rng() * Math.PI * 2,
      radius = Math.sqrt(rng()) * 27;
    const x = 2 + Math.cos(angle) * radius,
      y = Math.sin(angle) * radius;
    const clearing = Math.exp(-(((x - 1) / 5.3) ** 2) - (y / 2.8) ** 2);
    const density =
      0.62 + 0.2 * Math.sin(x * 0.28 + Math.sin(y * 0.2)) * Math.cos(y * 0.24);
    if (rng() > density * (1 - 0.96 * clearing)) continue;
    if (Math.hypot(x - 5, y + 7) < 5) continue;
    if ([-1.5, 0, 1.5].some((dy) => Math.hypot(x + 2, y - dy) < 1.2)) continue;
    if (clearings.some(c => Math.hypot(x - c.x, y - c.y) < c.radius)) continue;
    if (stems.some((tree) => Math.hypot(x - tree.x, y - tree.y) < 1.3))
      continue;
    const h = 3.4 + rng() ** 0.8 * 7.8;
    stems.push({
      x,
      y,
      h,
      r: 0.055 + h * (0.012 + rng() * 0.012),
      leanX: (rng() - 0.5) * h * 0.08,
      leanY: (rng() - 0.5) * h * 0.08,
    });
  }
  if (stems.length !== 108) throw Error("Could not place the forest stand");
  return stems;
}
