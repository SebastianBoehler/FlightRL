import type { Sample, Robot } from "./contracts";

export function historyPlot() {
  const canvas = document.getElementById("history") as HTMLCanvasElement;
  const signal = document.getElementById("signal-select") as HTMLSelectElement;
  let captures: Sample[] = [],
    selected: Robot | null = null,
    last: Sample | null = null;
  const value = (s: Sample) =>
    selected ? s.robots[selected.id]?.signals[signal.value] : null;
  function draw() {
    const caption = document.getElementById("history-label")!;
    const current = last && value(last);
    caption.textContent = `${selected?.label ?? "No robot"} · ${signal.selectedOptions[0]?.textContent ?? "No recorded signals"} · ${captures.length} samples · selected ${typeof current === "number" ? current.toFixed(3) : "unavailable"}`;
    canvas.setAttribute("aria-label", caption.textContent);
    if (!canvas.clientWidth || !canvas.clientHeight) return;
    const ratio = devicePixelRatio;
    canvas.width = Math.round(canvas.clientWidth * ratio);
    canvas.height = Math.round(canvas.clientHeight * ratio);
    const ctx = canvas.getContext("2d")!,
      w = canvas.clientWidth,
      h = canvas.clientHeight;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const values = captures
      .map(value)
      .filter((v): v is number => typeof v === "number" && Number.isFinite(v));
    if (!values.length) return;
    let min = Infinity,
      max = -Infinity;
    for (const v of values) {
      min = Math.min(min, v);
      max = Math.max(max, v);
    }
    const padding = Math.max(0.025, (max - min) * 0.05),
      low = min - padding,
      high = max + padding;
    const t0 = captures[0].time_s,
      t1 = captures.at(-1)!.time_s;
    const x = (t: number) =>
      58 + ((t - t0) / Math.max(0.1, t1 - t0)) * (w - 78);
    const y = (v: number) => h - 40 - ((v - low) / (high - low)) * (h - 72);
    ctx.font = "11px system-ui";
    for (let i = 0; i <= 4; i++) {
      const v = low + ((high - low) * i) / 4;
      ctx.strokeStyle = "#29332f";
      ctx.beginPath();
      ctx.moveTo(52, y(v));
      ctx.lineTo(w - 20, y(v));
      ctx.stroke();
      ctx.fillStyle = "#9ca9a4";
      ctx.fillText(v.toFixed(2), 4, y(v) + 4);
    }
    ctx.strokeStyle = "#9ec8ae";
    ctx.lineWidth = 2;
    ctx.beginPath();
    let gap = true;
    for (const s of captures) {
      const v = value(s);
      if (typeof v !== "number" || !Number.isFinite(v)) {
        gap = true;
        continue;
      }
      if (gap) ctx.moveTo(x(s.time_s), y(v));
      else ctx.lineTo(x(s.time_s), y(v));
      gap = false;
    }
    ctx.stroke();
    if (last) {
      ctx.strokeStyle = "#dcc798";
      ctx.beginPath();
      ctx.moveTo(x(last.time_s), 24);
      ctx.lineTo(x(last.time_s), h - 35);
      ctx.stroke();
    }
    ctx.fillStyle = "#9ca9a4";
    ctx.fillText(`${t0.toFixed(1)} s`, 58, h - 12);
    ctx.fillText(`${t1.toFixed(1)} s`, w - 65, h - 12);
  }
  signal.onchange = draw;
  const observer = new ResizeObserver(draw);
  observer.observe(canvas);
  window.addEventListener("pagehide", () => observer.disconnect(), {
    once: true,
  });
  return {
    select(robot: Robot) {
      selected = robot;
      signal.replaceChildren(
        ...robot.signals.map(([id, label]) => new Option(label, id)),
      );
      draw();
    },
    update(state: Sample, record = true) {
      last = state;
      if (record) captures.push(state);
      draw();
    },
    load(samples: Sample[]) {
      captures = samples;
      draw();
    },
    reset() {
      captures = [];
      last = null;
      draw();
    },
  };
}
