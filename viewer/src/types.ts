export interface Frame {
  tick: number;
  time_s: number;
  position: number[];
  quaternion: number[];
  estimate: number[];
  connected: boolean;
  mode: string;
  inspected: string[];
  discovered: string[];
  truth_inspected: number[];
  command: number[];
  clearance: number;
  airflow_samples?: number[][];
  settled_particles?: number[][];
  particles?: number[][];
  gust_m_s2?: number[];
  wind_m_s?: number[];
  mean_transmission?: number;
  dust_airborne?: number;
  dust_deposited?: number;
  dust_resuspensions?: number;
}
export interface Episode {
  name: string;
  scene: {
    room: number[];
    boxes: number[][];
    panels: number[][];
    identity: string;
    environment?: { surface_style: string; wind_m_s?: number[] };
  };
  records: Frame[];
  result: {
    status: string;
    coverage: number;
    collision: boolean;
    recovered: boolean;
    controller: string;
    events: { tick: number; type: string; marker?: string }[];
  };
  atlas: string;
  frameWidth: number;
  frameHeight: number;
  atlasColumns: number;
  policyHash: string | null;
}
export interface Index {
  episodes: { name: string; file: string }[];
  evaluation: {
    classicalCoverage: number;
    studentCoverage: number;
    studentPromoted: boolean;
    testLayouts: number;
    summary: string;
  };
}
