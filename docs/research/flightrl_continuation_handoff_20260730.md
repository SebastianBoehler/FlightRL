# FlightRL continuation handoff

Date: 2026-07-30

Read `docs/research/flightrl_thread_retrospective_20260730.md` before changing
training or flight authority. Follow
`docs/research/flightrl_experiment_control_protocol_20260730.md` for every new
checkpoint. The immediate goal is the first honest filmable semantic-student
flight, not another broad architecture iteration.

## Current candidate

Checkpoint:

`artifacts/puffer_fixed_door_d1_v59_fresh_control_bc1m/flightrl_fixed_door_d1_seed11_1048576.bin`

SHA-256:

`f676d12b9d37c27f4cc62f99beceec8f30e74c88be8564cb242c23755e202cce`

Corrected evaluation:

`artifacts/puffer_fixed_door_d1_v59_fresh_control_bc1m/flightrl_fixed_door_d1_seed11_1048576.reevaluation.json`

Authoritative metrics:

- 359 full-camera episodes;
- 79.11% success;
- 74.73% outside-FOV success;
- 0.56% collision;
- 1.53% masked-camera success;
- teacher: 93.55% success, 90.68% outside-FOV, 0.22% collision.

The original report's masked-camera result is stale. Use the reevaluation.

## Constraints

- only seed 11 has been evaluated;
- obstacles were disabled in this training configuration;
- fixed door category only;
- no v59 real camera shadow exists yet;
- no v59 live authority has been executed;
- native training used `max_rate_yaw = 4 rad/s` (229.18 deg/s), not the
  configured 70 deg/s;
- live v59 authority must remain clamped to 8 deg/s;
- the existing real door shadow is v53 and cannot approve v59;
- firmware must retain takeoff, stabilization, height/position hold, landing,
  and abort.

## Gate A: first semantic demo

Do not retrain before this gate unless the real v59 shadow reveals a contract
or direction defect.

1. Confirm no other task owns the drone, the battery is charged, the Flow Deck
   and AI Deck are stable, the path is clear, and the operator is present.
2. Run v59 in non-actuating shadow for at least 20 seconds with a real door
   entering and leaving the FOV.
3. Bind checkpoint, corrected evaluation, raw CSV, and summary by SHA.
4. Inspect frame age/dropouts, finite actions, phase transitions, confidence,
   yaw sign relative to centroid, and inference latency.
5. Build the v59 readiness artifact. Do not reuse the v53 shadow.
6. If readiness passes, fly yaw-only for at most 15 seconds at 0.5-0.8 m
   height, with translation hard-zero and yaw capped at 8 deg/s.
7. Start with the door 30-60 degrees outside FOV. Success is search, acquisition,
   centering, and stop without translation or abort.
8. Review the first log. Repeat once for filming only after it passes review.

Relevant entry points:

- `scripts/crazyflie_door_puffer_shadow.py`
- `scripts/build_fixed_door_live_readiness.py`
- `scripts/crazyflie_fixed_door_control.py`
- `src/flightrl/puffer4_door_control.py`
- `src/flightrl/puffer4_door_readiness.py`

## Gate B: bounded approach

Only after yaw-only success:

1. add at most 0.05 m/s forward authority;
2. cap travel to 0.10-0.20 m or three seconds;
3. require fresh centered evidence and a clear corridor;
4. command zero forward on stale evidence, low confidence, excessive flow, or
   target loss;
5. keep the operator abort and firmware landing path.

This is not an obstacle-exploration claim.

## Training v60

Run this as a parallel science path, not a prerequisite for Gate A:

1. fix the native yaw-scale contract and previous-executed-action encoding;
2. retain v59 perception initialization and fresh recurrent control;
3. compare pure BC against fresh-controller DAgger with identical samples,
   layouts, seeds, and evaluation;
4. use seeds 11, 23, and 47 first;
5. evaluate at least 1,000 held-out episodes per seed, stratified by initial
   visibility, topology family, light, occlusion, and distractor;
6. promote DAgger only if it adds at least five completion points without
   increasing collision above 3%;
7. add obstacle probability and camera/latency randomization in a separate
   experiment after the BC/DAgger comparison.

Do not combine yaw-scale repair, DAgger, more rooms, obstacles, and PPO in one
run. That would recreate the ambiguity of the earlier lineage.

## Stop rules

- No more deterministic-forward live repeats unless testing a hardware defect.
- No live semantic translation before a hash-bound v59 yaw-only pass.
- No obstacle-avoidance claim from v59.
- No shared language/target policy until fixed-door behavior is robust.
- No PPO continuation promoted without beating its bootstrap on the full gate.
- No room-count increase without equal samples per factor and held-out grammar.
- No repeatedly used real sequence described as a final holdout.
- No world model unless simple recurrence has a measured failure it can address.
- No raw motor or attitude-rate authority in this demo lane.

## Milestone definition

One continuous video and its bound log must show:

- stock firmware takeoff and stable hover;
- the door initially outside or near the edge of the FOV;
- v59 alone producing the yaw command;
- visible search, acquisition, centering, and stop;
- translation remaining zero;
- no stale-frame event, safety abort, or collision;
- checkpoint SHA, camera timing, detections, phase, proposed action, clamped
  action, and firmware state recorded.

That is the first defensible live semantic-student demo. The next milestones are
bounded approach, obstacle-rich training, shared target conditioning, and
on-edge deployment, in that order.
