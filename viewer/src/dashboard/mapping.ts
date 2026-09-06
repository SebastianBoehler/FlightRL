import { MapWorld } from "../reconstruction/world";
import type { Review } from "../reconstruction/types";
import type { Source, Sample } from "./contracts";
import { panel, el } from "./panel";
import { playback } from "./playback";
export async function start(source: Source) {
  const learned = source.params.backend === "lingbot",
    base = learned ? "/reconstruction/learned" : "/reconstruction";
  const response = await fetch(`${base}/review.json`);
  if (!response.ok) throw Error(`Reconstruction: HTTP ${response.status}`);
  const data: Review = await response.json(),
    ids = [...new Set(data.maps.map((m) => m.drone))];
  const ui = panel();
  ui.setup(
    ids.map((i) => ({
      id: String(i),
      label: `Camera ${i}`,
      signals: [
        ["x", "Estimated X"],
        ["y", "Estimated Y"],
        ["z", "Estimated Z"],
      ],
    })),
    [
      {
        id: String(ids[0]),
        label: "Selected camera · recorded RGB",
        width: 256,
        height: 192,
      },
    ],
  );
  const images = await Promise.all(
    Array.from({ length: Math.ceil(data.frames / 32) }, async (_, i) => {
      const img = new Image();
      img.src = `${base}/camera-${i}.jpg`;
      await img.decode();
      return img;
    }),
  );
  const world = new MapWorld(el("view")),
    actions = el("source-controls");
  actions.hidden = false;
  actions.innerHTML =
    '<h3>Reconstruction</h3><label>Method<select id="map-mode" aria-label="Reconstruction method"></select></label><label><input type="checkbox" id="truth"> Reference trajectory (evaluation only)</label><a id="export">Export point cloud</a>';
  const mode = el<HTMLSelectElement>("map-mode"),
    truth = el<HTMLInputElement>("truth");
  mode.replaceChildren(
    ...(learned
      ? [["rgb", "LingBot · RGB only"]]
      : [
          ["rgbd", "RGB-D · metric"],
          ["rgb", "RGB · arbitrary scale"],
        ]
    ).map(([value, label]) => new Option(label, value)),
  );
  truth.disabled = learned;
  let map = data.maps[0];
  let samples: Sample[] = [];
  function select() {
    const selected = data.maps.find(
      (m) => m.drone === Number(ui.selection()) && m.mode === mode.value,
    );
    if (!selected)
      throw Error("This camera has no reconstruction for the selected method");
    map = selected;
    const unit = map.mode === "rgbd" ? "m" : "map units";
    ids.forEach((id) =>
      ui.signals(
        String(id),
        ["x", "y", "z"].map((axis) => [
          axis,
          `Estimated ${axis.toUpperCase()} (${unit})`,
        ]),
      ),
    );
    world.load(map);
    samples = Array.from({ length: data.frames + 1 }, (_, frame) => ({
      time_s: Math.max(0, frame - 1) * data.dt,
      robots: Object.fromEntries(
        ids.map((id) => {
          const pose = frame
            ? data.maps.find((m) => m.drone === id && m.mode === mode.value)
                ?.poses[frame - 1]
            : null;
          return [
            String(id),
            {
              position: pose ?? undefined,
              signals: {
                x: pose?.[0] ?? null,
                y: pose?.[1] ?? null,
                z: pose?.[2] ?? null,
              },
            },
          ];
        }),
      ),
    }));
    ui.history(samples);
    el("pose-title").textContent = "Estimated camera position";
    el("pose-frame").textContent =
      map.mode === "rgbd"
        ? "Local camera frame · metres · orientation not recorded"
        : "Local camera frame · arbitrary map units";
    el<HTMLAnchorElement>("export").href =
      `${base}/drone-${map.drone}-${map.mode}.ply`;
    const feed = el("feeds").querySelector("figure")!;
    feed.dataset.robot = String(map.drone);
    feed.dataset.selected = "true";
    feed.querySelector("figcaption span")!.textContent =
      `Camera ${map.drone} · recorded RGB`;
    const inspect = feed.querySelector("button")!;
    inspect.setAttribute("aria-label", `Inspect camera ${map.drone}`);
    inspect.onclick = () => {
      if (el("inspector").hidden) el("toggle-inspector").click();
    };
    const meters = (v: number | null) =>
      v === null ? "unavailable" : `${v.toFixed(3)} m`;
    el("handover").textContent =
      `Full-run tracking ${(map.metrics.tracking_fraction * 100).toFixed(1)}% · trajectory RMSE ${meters(map.metrics.ate_rmse_m)} · mean surface error ${meters(map.metrics.surface_accuracy_m)} · observed surface coverage ${map.metrics.surface_coverage === null ? "unavailable" : (map.metrics.surface_coverage * 100).toFixed(1) + "%"} · ${data.result.mission.status}. ${data.result.registration.map((r) => `Camera ${r.drone}: ${r.accepted ? "accepted" : "rejected"} overlap (${r.inliers} matches)`).join(" · ")}`;
  }
  select();
  const clock = playback(
    samples.map((s) => s.time_s),
    (frame) => {
      const count = world.update(map, frame, truth.checked);
      ui.state(samples[frame], false);
      el("metrics").textContent =
        `${count.toLocaleString()} points · ${frame === 0 ? "Before first observation" : map.poses[frame - 1] ? "Pose available" : "Tracking unavailable; no pose interpolated"}`;
      const canvas = el<HTMLCanvasElement>("camera-0"),
        context = canvas.getContext("2d")!;
      if (!frame) context.clearRect(0, 0, 256, 192);
      else {
        const f = frame - 1;
        context.drawImage(
          images[Math.floor(f / 32)],
          (map.drone - 1) * 256,
          (f % 32) * 192,
          256,
          192,
          0,
          0,
          256,
          192,
        );
      }
      ui.captureLabel(
        frame
          ? `REPLAY · camera ${map.drone} · ${samples[frame].time_s.toFixed(3)} s · original RGB`
          : "Empty map · before first observation",
      );
    },
  );
  ui.onSelect(() => {
    select();
    clock.refresh();
  });
  mode.onchange = () => {
    select();
    clock.refresh();
  };
  truth.onchange = clock.refresh;
  el("overview").onclick = () => {
    world.load(map);
    clock.refresh();
  };
  el("focus-robot").onclick = () => world.focus(map, clock.index());
  el("status").textContent =
    "Maps remain in independent local frames. RGB uses arbitrary scale; metric error uses evaluator-only alignment. No loop closure. Missing tracking stays visible as gaps.";
  window.addEventListener("pagehide", () => world.dispose(), { once: true });
}
