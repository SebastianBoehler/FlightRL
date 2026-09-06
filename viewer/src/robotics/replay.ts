import type { RobotMessage } from "./types";

/** Decode all streams before displaying a new acquisition; ignore obsolete seeks. */
export function replayControls(send: (value: unknown) => void) {
  const timeline = document.getElementById("timeline") as HTMLInputElement;
  const latest = document.getElementById("live") as HTMLButtonElement;
  let recorded: Array<{ sequence: number; time_s: number }> = [];
  let requested = -1;
  timeline.oninput = () => {
    const entry = recorded[Number(timeline.value)];
    document.getElementById("replay-time")!.textContent = `${entry.time_s.toFixed(3)} s`;
    requested = entry.sequence;
    send({ type: "replay", sequence: requested });
    latest.disabled = false;
  };
  latest.onclick = () => {
    timeline.value = timeline.max;
    timeline.dispatchEvent(new Event("input"));
  };
  return {
    saved(captures: typeof recorded) {
      recorded = captures;
      timeline.max = String(Math.max(0, captures.length - 1));
      timeline.value = timeline.max;
      timeline.disabled = captures.length === 0;
      document.getElementById("replay-time")!.textContent = `${captures.length} captures saved · drag to replay`;
    },
    async show(message: Extract<RobotMessage, { type: "replay" }>, names: string[]) {
      if (message.state.sequence !== requested) return false;
      const images = await Promise.all(names.map(async (name) => {
        const img = new Image();
        img.src = `data:image/png;base64,${message.images[name]}`;
        await img.decode();
        return img;
      }));
      if (message.state.sequence !== requested) return false;
      images.forEach((img, i) => {
        const canvas = document.getElementById(`camera-${i}`) as HTMLCanvasElement;
        canvas.getContext("2d")!.drawImage(img, 0, 0);
        canvas.setAttribute("aria-label", `${names[i]} recorded raw RGB acquisition`);
      });
      return true;
    },
  };
}
