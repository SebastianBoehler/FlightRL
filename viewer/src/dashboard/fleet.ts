import { chapters } from "./chapters";
import { prepareForestTextures } from "../forest/scanned-materials";
import { FleetWorld } from "../fleet/world";
import type { FleetReplay } from "../fleet/types";
import type { Source, Sample } from "./contracts";
import { panel, el } from "./panel";
import { playback } from "./playback";
export async function start(source: Source) {
  const response = await fetch(`/fleet/${source.params.mission}.json`);
  if (!response.ok) throw Error(`Fleet recording: HTTP ${response.status}`);
  const replay: FleetReplay = await response.json();
  if (
    replay.records.some(
      (f) => f.positions.length !== 3 || f.quaternions.length !== 3,
    )
  )
    throw Error("Fleet recording must contain three robot poses per frame");
  const ui = panel(),
    ids = ["drone-1", "drone-2", "drone-3"];
  ui.setup(
    ids.map((id, i) => ({
      id,
      label: `Drone ${i + 1}${replay.provenance.roles?.[i] ? ` · ${replay.provenance.roles[i]}` : ""}`,
      signals: [
        ["height", "Body height (m)"],
        ["distance", "Goal distance (m)"],
      ],
    })),
    ids.map((id, i) => ({
      id,
      label: `Drone ${i + 1} · ${replay.sensor_atlas ? "recorded policy RGB" : "pose re-render"}`,
      width: replay.sensor_atlas ? 64 : 256,
      height: replay.sensor_atlas ? 48 : 192,
    })),
  );
  let atlas: ImageBitmap | null = null;
  if (replay.sensor_atlas) {
    const response = await fetch(replay.sensor_atlas);
    if (!response.ok) throw Error(`Sensor recording: HTTP ${response.status}`);
    atlas = await createImageBitmap(await response.blob());
    if (atlas.width !== 64 * 3 || atlas.height !== 48 * replay.records.length)
      throw Error("Sensor atlas does not match replay");
  }
  ids.forEach((_, i) => {
    const label = document.createElement("span");
    label.id = `drone-label-${i}`;
    label.className = "drone-label";
    label.textContent = `Drone ${i + 1}`;
    el("view").append(label);
  });
  if (replay.provenance.family === "forest") await prepareForestTextures();
  const world = new FleetWorld(replay, [
    el("view"),
    ...ids.map((_, i) => el(`camera-${i}`)),
  ]);
  await world.renderer.init();
  const samples: Sample[] = replay.records.map((f) => ({
    time_s: f.time_s,
    robots: Object.fromEntries(
      ids.map((id, i) => {
        const q = f.quaternions[i];
        return [
          id,
          {
            position: f.positions[i],
            yaw:
              (Math.atan2(
                2 * (q[0] * q[3] + q[1] * q[2]),
                1 - 2 * (q[2] ** 2 + q[3] ** 2),
              ) *
                180) /
              Math.PI,
            signals: {
              height: f.positions[i][2],
              distance: Math.hypot(
                ...f.positions[i].map((v, j) => v - f.goals[i][j]),
              ),
            },
          },
        ];
      }),
    ),
  }));
  ui.history(samples);
  ui.onSelect((id) => {
    world.selected = ids.indexOf(id);
  });
  el("overview").onclick = () => world.overview();
  el("focus-robot").onclick = () => {
    world.follow = true;
    clock.refresh();
  };
  el("status").textContent = [
    replay.provenance.vehicle,
    replay.provenance.camera,
    replay.provenance.communication,
    replay.provenance.scope,
    replay.provenance.evaluation,
  ]
    .filter(Boolean)
    .join(" · ");
  el("pose-title").textContent = "Body pose";
  const clock = playback(
    replay.records.map((f) => f.time_s),
    (index) => {
      const f = replay.records[index];
      world.update(f, index);
      ui.state(samples[index], false);
      if (atlas)
        ids.forEach((_, i) =>
          el<HTMLCanvasElement>(`camera-${i}`)
            .getContext("2d")!
            .drawImage(atlas!, i * 64, index * 48, 64, 48, 0, 0, 64, 48),
        );
      ui.captureLabel(
        `REPLAY · ${f.time_s.toFixed(3)} s · ${atlas ? "original recorded policy RGB" : "rendered at recorded poses; training images not retained"}`,
      );
      el("metrics").textContent =
        `${f.completed.filter(Boolean).length}/3 robots complete · ${f.task_done ? `${f.task_done.filter(Boolean).length}/${f.task_done.length} tasks complete` : replay.result.controller}`;
      el("handover").textContent =
        `${index === replay.records.length - 1 ? replay.result.status + " · " : ""}${
          replay.events
            ?.filter((e) => e.time_s <= f.time_s)
            .slice(-3)
            .map((e) => e.text)
            .join(" · ") ?? "No event log in this recording"
        }`;
    },
  );
  chapters(
    (replay.events ?? []).map((e) => ({
      label: e.text,
      index: replay.records.findIndex((f) => f.time_s >= e.time_s),
    })),
    clock.seek,
  );
  let busy = false,
    last = 0;
  world.renderer.setAnimationLoop(async (now) => {
    if (busy || document.hidden || now - last < 1000 / 30) return;
    busy = true;
    last = now;
    try {
      await world.draw();
    } catch (error) {
      el("error").textContent = String(error);
      world.renderer.setAnimationLoop(null);
    } finally {
      busy = false;
    }
  });
  window.addEventListener(
    "pagehide",
    () => {
      atlas?.close();
      world.dispose();
    },
    { once: true },
  );
}
