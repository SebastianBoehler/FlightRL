from __future__ import annotations

import argparse

from flightrl.sixdof.command_replay import load_box_room, load_csv, replay_velocity_commands, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay Crazyflie velocity/yaw-rate logs through the 6-DoF simulator")
    parser.add_argument("--input", required=True)
    parser.add_argument("--room-report", default=None)
    parser.add_argument("--output", default="artifacts/trajectories/crazyflie_command_replay.csv")
    parser.add_argument("--normalized-real-output", default=None)
    parser.add_argument("--raw-origin", action="store_true")
    parser.add_argument("--override-z-m", type=float, default=None, help="Use an explicit derived height when logged z is unreliable.")
    parser.add_argument("--hold-z-m", type=float, default=None, help="Hold an explicit replay height instead of using logged vz_m_s.")
    parser.add_argument("--velocity-gain", type=float, default=2.5)
    parser.add_argument("--yawrate-scale", type=float, default=1.0)
    parser.add_argument("--command-frame", choices=("body", "world"), default="body")
    parser.add_argument("--yaw-source", choices=("logged", "sim"), default="logged")
    parser.add_argument("--vx-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--vy-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--max-dt-s", type=float, default=0.08)
    args = parser.parse_args()

    room = load_box_room(args.room_report)
    sim_rows, real_rows = replay_velocity_commands(
        load_csv(args.input),
        room=room,
        normalize_origin=not args.raw_origin,
        override_z_m=args.override_z_m,
        hold_z_m=args.hold_z_m,
        velocity_gain=args.velocity_gain,
        yawrate_scale=args.yawrate_scale,
        command_frame=args.command_frame,
        yaw_source=args.yaw_source,
        vx_sign=args.vx_sign,
        vy_sign=args.vy_sign,
        max_dt_s=args.max_dt_s,
    )
    write_csv(args.output, sim_rows)
    if args.normalized_real_output:
        write_csv(args.normalized_real_output, real_rows)
    print(f"wrote {len(sim_rows)} simulated rows to {args.output}")
    if args.normalized_real_output:
        print(f"wrote {len(real_rows)} normalized real rows to {args.normalized_real_output}")


if __name__ == "__main__":
    main()
