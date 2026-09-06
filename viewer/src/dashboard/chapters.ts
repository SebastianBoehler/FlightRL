import { el } from "./panel";
/** Event navigation uses existing source timestamps, never guessed playback offsets. */
export function chapters(
  events: Array<{ label: string; index: number }>,
  seek: (index: number) => void,
) {
  if (!events.length) return;
  const host = el("source-controls");
  host.hidden = false;
  const label = document.createElement("label");
  label.textContent = "Jump to event";
  const select = document.createElement("select");
  select.setAttribute("aria-label", "Replay event");
  select.add(new Option("Select an event", ""));
  events
    .filter((e) => e.index >= 0)
    .forEach((e) => select.add(new Option(e.label, String(e.index))));
  select.onchange = () => {
    if (select.value) seek(Number(select.value));
  };
  label.append(select);
  host.append(label);
}
