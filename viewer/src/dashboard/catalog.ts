import { missions } from "../missions";
import type { Index } from "../types";
import type { Source } from "./contracts";

export async function catalog(): Promise<Source[]> {
  const response = await fetch("/data/index.json");
  if (!response.ok) throw Error(`Run catalog: HTTP ${response.status}`);
  const index: Index = await response.json();
  return [
    ...["production", "power"].map((site) => ({
      id: site,
      label: site === "power" ? "Power campus" : "Production campus",
      group: "Live simulation",
      adapter: "robotics" as const,
      params: { site },
    })),
    {
      id: "live-forest",
      label: "Forest · dust & flight",
      group: "Live simulation",
      adapter: "realism",
      params: {},
    },
    ...index.episodes.map((e) => ({
      id: e.file,
      label: e.name.replaceAll("_", " "),
      group: "Recorded single robot",
      adapter: "single" as const,
      params: { episode: e.file },
    })),
    {
      id: "forest-quality",
      label: "Forest · detailed scene",
      group: "Recorded single robot",
      adapter: "forest",
      params: {},
    },
    ...missions
      .filter((m) => m.id !== "single")
      .map((m) => ({
        id: m.id,
        label: m.label.replace("3 drones · ", ""),
        group: "Recorded fleet",
        adapter: "fleet" as const,
        params: { mission: m.id },
      })),
    {
      id: "rgbd-map",
      label: "RGB-D / RGB baseline",
      group: "Reconstruction",
      adapter: "mapping",
      params: {},
    },
    {
      id: "learned-map",
      label: "LingBot RGB trial",
      group: "Reconstruction",
      adapter: "mapping",
      params: { backend: "lingbot" },
    },
  ];
}
export function selectedSource(sources: Source[]): Source {
  const params = new URLSearchParams(location.search);
  let id = params.get("source");
  if (!id) {
    const path = location.pathname;
    if (path === "/robotics.html") id = params.get("site") ?? "production";
    else if (path === "/realism.html") id = "live-forest";
    else if (path === "/forest.html") id = "forest-quality";
    else if (path === "/mapping.html")
      id = params.get("backend") === "lingbot" ? "learned-map" : "rgbd-map";
    else if (
      path === "/fleet.html" ||
      params.has("mission") ||
      params.has("run")
    ) {
      id = params.get("mission") ?? params.get("run") ?? "cooperative";
      if (id === "single")
        id =
          params.get("episode") ??
          sources.find((s) => s.adapter === "single")!.id;
    } else id = "production";
  }
  const source = sources.find((s) => s.id === id);
  if (!source) throw Error(`Unknown environment or run: ${id}`);
  return source;
}
export function sourceURL(source: Source) {
  return `/?${new URLSearchParams({ source: source.id, ...source.params })}`;
}
