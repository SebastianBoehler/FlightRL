/** Source-native values; absent measurements remain unavailable, never synthesized. */
export interface RobotSample {
  position?: number[];
  yaw?: number;
  signals: Record<string, number | null>;
}
export interface Sample {
  time_s: number;
  robots: Record<string, RobotSample>;
}
export interface Robot {
  id: string;
  label: string;
  signals: Array<[string, string]>;
}
export interface Source {
  id: string;
  label: string;
  group: string;
  adapter: "robotics" | "realism" | "single" | "fleet" | "mapping" | "forest";
  params: Record<string, string>;
}
