import { mountShell } from "./dashboard/shell";
import { catalog, selectedSource, sourceURL } from "./dashboard/catalog";
import "./dashboard/style.css";
import "./dashboard/responsive.css";
import "./dashboard/sources.css";

mountShell();
async function start() {
  const sources = await catalog();
  const select = document.getElementById("source-select") as HTMLSelectElement;
  select.replaceChildren();
  for (const group of new Set(sources.map((s) => s.group))) {
    const options = document.createElement("optgroup");
    options.label = group;
    options.append(
      ...sources
        .filter((s) => s.group === group)
        .map((s) => new Option(s.label, s.id)),
    );
    select.append(options);
  }
  select.disabled = false;
  select.onchange = () =>
    location.assign(sourceURL(sources.find((s) => s.id === select.value)!));
  const source = selectedSource(sources);
  select.value = source.id;
  document.getElementById("metrics")!.textContent = `Loading ${source.label}…`;
  document.getElementById("status")!.textContent =
    `Loading ${source.group.toLowerCase()} source…`;
  history.replaceState(null, "", sourceURL(source));
  document.querySelector("h1")!.textContent = source.label;
  document.querySelector("header p")!.textContent = source.group;
  document.querySelector(".environment-label")!.textContent =
    source.group === "Live simulation" ? "Simulation" : "Recorded";
  document.getElementById("reset")!.onclick = () => location.reload();
  switch (source.adapter) {
    case "robotics":
      await (await import("./robotics/main")).start();
      break;
    case "realism":
      await (await import("./realism/main")).start();
      break;
    case "single":
      await (await import("./dashboard/single")).start(source);
      break;
    case "fleet":
      await (await import("./dashboard/fleet")).start(source);
      break;
    case "mapping":
      await (await import("./dashboard/mapping")).start(source);
      break;
    case "forest":
      await (await import("./forest/main")).start();
      break;
  }
}
start().catch((error) => {
  document.getElementById("error")!.textContent = String(error);
  document.getElementById("metrics")!.textContent = "Source could not load";
  document.getElementById("status")!.textContent =
    "Select another source or restart after resolving the error.";
});
