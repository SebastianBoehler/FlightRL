import type { Handover } from "./inspection-status";
import type { WorldState } from "../realism/types";
import type { SceneDescription } from "./scene";

export interface RobotState {
  time_s: number;
  sequence: number;
  bodies: { positions: number[][]; quaternions: number[][] };
  camera: Pick<WorldState, "positions" | "quaternions">;
  proprio: number[][];
  encoder: number[];
  imu: number[];
  capture_time_ns: number;
  camera_poses: Array<{position_m: number[]; rotation: number[]}>;
  arm: null | {
    position_rad: number[]; velocity_rad_s: number[]; effort_nm: number[];
    actuator_names: string[]; control: number[]; control_limits: number[][];
    actuator_force: number[];
  };
}
export type RobotMessage =
  | {
      type: "scene";
      description: SceneDescription;
      state: RobotState;
      label: string;
    }
  | { type: "state"; state: RobotState }
  | { type: "capture"; id: number; state: RobotState }
  | {
      type: "metrics";
      count: number;
      status: string;
      done?: boolean;
      handover?: Handover | null;
      sensor_valid?: boolean[];
    }
  | { type: "saved"; path: string; captures: Array<{sequence: number; time_s: number}> }
  | { type: "replay"; state: RobotState; images: Record<string, string> }
  | { type: "error"; message: string };
