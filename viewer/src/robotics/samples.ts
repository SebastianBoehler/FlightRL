import type { RobotState } from "./types";
import type { Robot, Sample } from "../dashboard/contracts";
export function robots(ids: string[]): Robot[] {
  return ids.map((id) => ({
    id,
    label: id === "arm" ? "xArm7 · wrist camera" : id,
    signals:
      id === "arm"
        ? [
            ["position", "Joint 1 position (rad)"],
            ["velocity", "Joint 1 velocity (rad/s)"],
            ["effort", "Joint 1 effort (Nm)"],
          ]
        : [
            ["height", "Camera height (m)"],
            ["speed", "Measured speed (m/s)"],
          ],
  }));
}
export function sample(state: RobotState, ids: string[]): Sample {
  return {
    time_s: state.time_s,
    robots: Object.fromEntries(
      ids.map((id, i) => {
        const p = state.camera_poses[i],
          a = state.arm;
        return [
          id,
          {
            position: p.position_m,
            yaw: (Math.atan2(p.rotation[3], p.rotation[0]) * 180) / Math.PI,
            signals: (id === "arm"
              ? {
                  position: a?.position_rad[0] ?? null,
                  velocity: a?.velocity_rad_s[0] ?? null,
                  effort: a?.effort_nm[0] ?? null,
                }
              : {
                  height: p.position_m[2],
                  speed: state.proprio[i]
                    ? Math.hypot(...state.proprio[i].slice(0, 3))
                    : null,
                }) as Record<string, number | null>,
          },
        ];
      }),
    ),
  };
}
