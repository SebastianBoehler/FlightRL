# Room-scene research boundary

This is an offline simulator/data idea, not an implemented mapping-to-flight
workflow. The active repository has room/ranger projection helpers and
simulation scene types, but no reviewed command that turns a scan into learned
live control.

## Useful narrow artifact

A future `RoomScene` should be a small versioned JSON file containing measured
frame identity, bounds, uncertainty, and optionally reviewed obstacles/targets.
Every value needs explicit units and reference frames. Source telemetry,
hardware/firmware/config identity, cleaning parameters, and hashes must be
bound to the artifact.

Start with bounds only. Sparse range hits and Flow-derived pose can support a
coarse room approximation, but they do not establish dense furniture geometry,
semantic target identity, or a drift-free navigation frame.

## Valid desktop uses

- replay and visualize the measured points and fitted bounds;
- instantiate a held-out `BoxRoom`-style simulator case;
- compare native and MuJoCo ray/geometry semantics;
- test how a desktop teacher or simulation policy responds to uncertainty;
- generate regression fixtures for frame, sign, unit, and clearance bugs.

Do not place measured scenes in the training split when they are intended as
physical holdouts. `num_envs` is vectorized independent simulation, not several
drones sharing one room.

## Required checks before any future extension

1. reject missing/nonfinite rows, poses, ranges, and bounds;
2. make sensor origin, axis convention, angle units, range limits, and scan
   frame explicit;
3. report point rejection and uncertainty rather than silently dropping data;
4. validate known-distance fixtures and loop-closure/drift behavior;
5. preserve physical source data independently from regenerable scene output;
6. keep scene artifacts as desktop evidence with `authority=none`.

Mapping accuracy and simulator replay would still not authorize hardware
commands. Any later map-assisted onboard feature would need a new versioned
edge/mission contract, budget, parity tests, independent STM32 safety handling,
and its own staged promotion process.
