import { historyPlot } from "./history";
import type { Robot, Sample } from "./contracts";
export const el = <T extends HTMLElement = HTMLElement>(id: string) =>
  document.getElementById(id) as T;
export function panel() {
  const history = historyPlot(),
    select = el<HTMLSelectElement>("robot-select");
  let robots: Robot[] = [],
    last: Sample | null = null;
  const listeners: Array<(id: string) => void> = [];
  function pose() {
    const robot = last?.robots[select.value];
    const values = [...(robot?.position ?? [null, null, null]), robot?.yaw];
    el("robot-pose")
      .querySelectorAll("dd")
      .forEach(
        (node, i) =>
          (node.textContent =
            typeof values[i] === "number"
              ? values[i]!.toFixed(i === 3 ? 1 : 3)
              : "—"),
      );
  }
  function choose() {
    el("feeds")
      .querySelectorAll<HTMLElement>("figure")
      .forEach(
        (f) => (f.dataset.selected = String(f.dataset.robot === select.value)),
      );
    const robot = robots.find((r) => r.id === select.value);
    if (robot) history.select(robot);
    pose();
    listeners.forEach((fn) => fn(select.value));
  }
  select.onchange = choose;
  return {
    setup(
      items: Robot[],
      feeds: Array<{
        id: string;
        label: string;
        width: number;
        height: number;
        canvasId?: string;
      }>,
    ) {
      robots = items;
      last = null;
      history.reset();
      el("feeds").style.setProperty(
        "--feed-count",
        String(Math.min(3, feeds.length)),
      );
      select.replaceChildren(...robots.map((r) => new Option(r.label, r.id)));
      el("feeds").replaceChildren(
        ...feeds.map((f, i) => {
          const figure = document.createElement("figure");
          figure.dataset.robot = f.id;
          const canvas = document.createElement("canvas");
          canvas.id = f.canvasId ?? `camera-${i}`;
          canvas.width = f.width;
          canvas.height = f.height;
          canvas.setAttribute("aria-label", f.label);
          const caption = document.createElement("figcaption"),
            label = document.createElement("span"),
            button = document.createElement("button");
          label.textContent = f.label;
          button.textContent = "Inspect";
          button.setAttribute("aria-label", `Inspect ${f.id}`);
          button.onclick = () => {
            select.value = f.id;
            choose();
            if (el("inspector").hidden) el("toggle-inspector").click();
          };
          caption.append(label, button);
          figure.append(canvas, caption);
          return figure;
        }),
      );
      choose();
    },
    state(sample: Sample, record = true) {
      last = sample;
      pose();
      history.update(sample, record);
    },
    history: history.load,
    signals(id: string, signals: Robot["signals"]) {
      const robot = robots.find((r) => r.id === id);
      if (robot) {
        robot.signals = signals;
        if (select.value === id) history.select(robot);
      }
    },
    selection: () => select.value,
    onSelect(fn: (id: string) => void) {
      listeners.push(fn);
    },
    captureLabel(text: string) {
      el("capture-time").textContent = text;
    },
  };
}
