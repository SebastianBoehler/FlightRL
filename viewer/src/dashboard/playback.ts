import { el } from "./panel";
/** One replay clock uses source sample times; seeking never invents intermediate observations. */
export function playback(times: number[], draw: (index: number) => void) {
  if (
    !times.length ||
    times.some((t, i) => !Number.isFinite(t) || (i > 0 && t < times[i - 1]))
  )
    throw Error("Replay timestamps must be finite and ordered");
  const slider = el<HTMLInputElement>("timeline"),
    play = el<HTMLButtonElement>("pause"),
    speed = el<HTMLSelectElement>("speed");
  let index = 0,
    playing = false,
    elapsed = times[0],
    previous = 0;
  const show = () => {
    slider.value = String(index);
    draw(index);
    el("replay-time").textContent =
      `${times[index].toFixed(2)} / ${times.at(-1)!.toFixed(2)} s`;
  };
  const stop = () => {
    playing = false;
    play.textContent = "Play";
  };
  function seek(i: number) {
    stop();
    index = i;
    elapsed = times[i];
    show();
  }
  slider.disabled = play.disabled = false;
  slider.max = String(times.length - 1);
  slider.step = "1";
  speed.hidden = false;
  play.textContent = "Play";
  play.onclick = () => {
    if (playing) {
      stop();
      return;
    }
    if (index === times.length - 1) seek(0);
    playing = true;
    previous = performance.now();
    play.textContent = "Pause";
  };
  slider.oninput = () => seek(Number(slider.value));
  el("reset").textContent = "Start";
  el("reset").onclick = () => seek(0);
  const end = el<HTMLButtonElement>("live");
  end.disabled = false;
  end.textContent = "End";
  end.onclick = () => seek(times.length - 1);
  let request = 0;
  function tick(now: number) {
    if (playing && !document.hidden) {
      elapsed += Math.min(0.1, (now - previous) / 1000) * Number(speed.value);
      let next = index;
      while (next + 1 < times.length && times[next + 1] <= elapsed) next++;
      if (next !== index) {
        index = next;
        show();
      }
      if (index === times.length - 1) stop();
    }
    previous = now;
    request = requestAnimationFrame(tick);
  }
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stop();
  });
  window.addEventListener("pagehide", () => cancelAnimationFrame(request), {
    once: true,
  });
  show();
  request = requestAnimationFrame(tick);
  return { seek, index: () => index, refresh: show };
}
