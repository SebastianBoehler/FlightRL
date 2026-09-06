export interface FleetFrame {
  time_s: number;
  positions: number[][];
  quaternions: number[][];
  goals: number[][];
  completed: boolean[];
  active?: boolean[];
  assignments?: number[];
  task_done?: boolean[];
  task_found?: boolean[];
  task_owners?: number[];
  completed_by?: number[][];
}
export interface FleetReplay {
  records: FleetFrame[];
  sensor_atlas?: string;
  tasks?: number[][];
  events?: {time_s:number; type:string; drone:number; task?:number; text:string}[];
  scene: { boxes: number[][]; room: number[]; panels?: number[][] };
  result: { status: string; controller: string; messages_delivered?: number };
  provenance: { family: string; mission?:string; roles?:string[]; scope?:string; seed: number; replay_sha256?: string; evaluation?:string; camera: string; communication: string; vehicle: string; dimensions: number[] };
}
