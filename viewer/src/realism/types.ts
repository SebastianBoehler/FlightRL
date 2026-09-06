import type { DroneReference } from "../models/drone";
export interface RigidBodySpec {
  id: string;
  position: number[];
  quaternion: number[];
  halfExtents: number[];
  mass: number;
  vehicle?: DroneReference["id"];
  model?: DroneReference;
}
export interface SharedScene {
  schema: "flightrl.shared_forest.v1";
  units: "m";
  up: "z";
  quaternionOrder: "xyzw";
  wind_m_s: number[];
  vertices: string;
  indices: string;
  triangleCount: number;
  bodies: RigidBodySpec[];
  materialAssets: unknown;
}
export interface WorldState {
  wind_m_s: number[];
  sequence: number;
  time_s: number;
  positions: number[][];
  quaternions: number[][];
  velocities: number[][];
  rates: number[][];
  mode: string;
  contacts: number;
  notice?: string;
  particles?: number[][];
  particleKinds?: number[];
}
export interface CameraRequest {
  type: "capture";
  id: number;
  state: WorldState;
}
