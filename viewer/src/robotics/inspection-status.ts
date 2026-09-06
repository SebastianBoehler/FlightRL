export interface Handover {
  asset_id: number;
  followup_marker: number;
  observed_signal: number;
  estimated_position_m: number[];
  position_variance_m2: number;
  capture_time_s: number;
}
export function inspectionStatus(
  handover: Handover | null | undefined,
  valid: boolean[] | undefined,
  assets: Array<{ id: number; asset?: string }>,
) {
  const panel = document.getElementById("handover")!;
  if (handover) {
    const asset =
      assets.find((a) => a.id === handover.asset_id)?.asset ??
      `Asset ${handover.asset_id}`;
    const next =
      assets.find((a) => a.id === handover.followup_marker)?.asset ??
      `Asset ${handover.followup_marker}`;
    panel.textContent = `${asset}: ${handover.observed_signal === 1 ? "FAULT indicator observed" : "normal indicator observed"} · ${next} verification requested · location estimate ${handover.estimated_position_m.map((v) => v.toFixed(1)).join(", ")} m · modeled uncertainty σ ${Math.sqrt(handover.position_variance_m2).toFixed(2)} m`;
  } else
    panel.textContent =
      "Waiting for the drone’s equipment reading and inspection handover.";
  valid?.forEach((available, i) => {
    const canvas = document.getElementById(`camera-${i}`)!;
    canvas.setAttribute(
      "aria-label",
      available
        ? "Raw RGB acquisition; controller receives delayed measurements"
        : "Raw RGB acquisition; latest controller measurement is unavailable",
    );
  });
}
