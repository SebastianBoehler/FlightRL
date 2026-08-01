# Navigation Autonomy

FlightRL's navigation package contains target-aware room generation and a pure
single-drone mission state machine. Policy inputs and outputs are defined only
by the edge-v3 contract; the former range/setpoint checkpoint benchmark was
removed because it described an incompatible action and observation surface.

## Mission State Machine

`flightrl.navigation.mission` defines the high-level single-drone mission flow:

```text
preflight -> takeoff -> search -> navigate -> recover -> hold -> land -> abort
```

The state machine is pure Python and has no hardware side effects. Live runners
can use it later to decide which controller or policy is active while preserving
phase-specific speed/yaw limits.
