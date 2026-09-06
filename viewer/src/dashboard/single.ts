import { chapters } from "./chapters";
import { World } from "../world";
import type { Episode } from "../types";
import type { Source, Sample } from "./contracts";
import { panel, el } from "./panel";
import { playback } from "./playback";
export async function start(source: Source) {
  const response = await fetch(`/data/${source.params.episode}`);
  if (!response.ok) throw Error(`Episode: HTTP ${response.status}`);
  const episode: Episode = await response.json();
  const atlas = new Image();
  atlas.src = `/data/${episode.atlas}`;
  await atlas.decode();
  const ui = panel();
  ui.setup(
    [
      {
        id: "drone",
        label: "Drone",
        signals: [
          ["height", "Body height (m)"],
          ["clearance", "Clearance (m)"],
          ["wind", "Wind speed (m/s)"],
          ["dust", "Airborne dust parcels"],
        ],
      },
    ],
    [
      {
        id: "drone",
        label: "Onboard · recorded RGB",
        width: episode.frameWidth,
        height: episode.frameHeight,
        canvasId: "camera",
      },
      {
        id: "drone",
        label: "Operator · recorded link availability",
        width: episode.frameWidth,
        height: episode.frameHeight,
        canvasId: "operator-camera",
      },
    ],
  );
  const samples: Sample[] = episode.records.map((f) => ({
    time_s: f.time_s,
    robots: {
      drone: {
        position: f.position,
        yaw:
          (Math.atan2(
            2 *
              (f.quaternion[0] * f.quaternion[3] +
                f.quaternion[1] * f.quaternion[2]),
            1 - 2 * (f.quaternion[2] ** 2 + f.quaternion[3] ** 2),
          ) *
            180) /
          Math.PI,
        signals: {
          height: f.position[2],
          clearance: f.clearance,
          wind: f.wind_m_s ? Math.hypot(...f.wind_m_s) : null,
          dust: f.dust_airborne ?? null,
        },
      },
    },
  }));
  ui.history(samples);
  // Keep source pixels even when the scene uses the detailed forest renderer.
  const world = new World(el("view"), false);
  await world.start();
  world.load(episode);
  const actions = el("source-controls");
  actions.hidden = false;
  actions.innerHTML =
    '<h3>Scene controls</h3><button id="airflow-toggle" aria-pressed="true">Airflow arrows</button><button id="camera-pose">Camera pose</button><a href="/data/evaluation.json" target="_blank">Evaluation report</a>';
  el("airflow-toggle").onclick = () => {
    world.airflow.visible = !world.airflow.visible;
    el("airflow-toggle").setAttribute(
      "aria-pressed",
      String(world.airflow.visible),
    );
  };
  el("overview").onclick = () => {
    world.mode = "Overview";
    world.overview();
  };
  el("focus-robot").onclick = () => {
    world.mode = "Follow drone";
    clock.refresh();
  };
  el("camera-pose").onclick = () => {
    world.mode = "Camera pose";
    clock.refresh();
  };
  el("status").textContent =
    `${episode.result.controller} · ${episode.result.status} · original source camera pixels.${episode.scene.environment?.surface_style === "forest" ? " Forest detail is a visualization; collision evaluation used the original scene." : ""}`;
  el("pose-title").textContent = "Body pose";
  const clock = playback(
    episode.records.map((f) => f.time_s),
    (index) => {
      const f = episode.records[index];
      world.update(f, index, atlas);
      ui.state(samples[index], false);
      const camera = el<HTMLCanvasElement>("camera"),
        operator = el<HTMLCanvasElement>("operator-camera");
      camera.getContext("2d")!.drawImage(world.atlasCanvas, 0, 0);
      operator
        .getContext("2d")!
        .clearRect(0, 0, operator.width, operator.height);
      if (f.connected)
        operator.getContext("2d")!.drawImage(world.atlasCanvas, 0, 0);
      operator.nextElementSibling!.querySelector("span")!.textContent =
        f.connected
          ? "Operator · received at source time"
          : "Operator · link unavailable";
      ui.captureLabel(
        `REPLAY · ${f.time_s.toFixed(3)} s · original recorded RGB`,
      );
      el("metrics").textContent =
        `${f.mode.replaceAll("_", " ")} · ${f.truth_inspected.length}/${episode.scene.panels.length} inspected`;
      el("handover").textContent =
        `${f.connected ? "Operator link connected" : "Operator link lost"} · ${episode.result.events
          .filter((e) => e.tick <= f.tick)
          .slice(-2)
          .map((e) => e.type)
          .join(" · ")}`;
    },
  );
  chapters(
    episode.result.events.map((e) => ({
      label: e.type,
      index: episode.records.findIndex((f) => f.tick >= e.tick),
    })),
    clock.seek,
  );
}
