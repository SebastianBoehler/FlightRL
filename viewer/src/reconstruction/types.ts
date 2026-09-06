export type Point = [number, number, number];
export interface MapData {
  drone: number;
  mode: "rgb" | "rgbd";
  points: [Point, Point, number][];
  poses: (Point | null)[];
  truth: Point[];
  metrics: {
    tracking_fraction: number;
    ate_rmse_m: number | null;
    surface_accuracy_m: number | null;
    surface_coverage: number | null;
  };
}
export interface Review {
  backend?: string;
  warmup?: number;
  seed: number;
  frames: number;
  dt: number;
  maps: MapData[];
  result: {
    mission: { status: string; mission_time_s: number };
    registration: { drone: number; accepted: boolean; inliers: number }[];
  };
}
