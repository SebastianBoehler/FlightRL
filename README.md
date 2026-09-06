# FlightRL

FlightRL is a local-first autonomy co-design stack for compiling vehicle,
terrain, sensor, and mission descriptions into fast simulations, bounded
policies, and reproducible edge artifacts. The current hardware lane targets a
small target-conditioned Crazyflie AI Deck policy. Its actor proposes bounded
velocity and yaw-rate setpoints; the STM32 estimator, safety layer, stabilizer,
and motor mixer remain authoritative.

The forward architecture separates a general typed mission supervisor from the
learned controller. A versioned policy-I/O contract can describe contiguous raw
camera, IMU, actuator, electrical, thermal, typed-goal, embodiment, and neighbor
signals with only explicit affine calibration, then select velocity, body-rate,
or direct per-motor action heads. Language stays in the low-rate mission
compiler. These new contracts are simulation/training foundations; they do not
make the current hardware lane direct-motor capable.

## Current status

- The canonical target is `aideck-navigation-policy-v3`: one 64x48 gray4 frame,
  19 telemetry values, one of three target IDs, and four bounded setpoint
  proposals.
- The PyTorch reference actor has 17,602 parameters and an estimated 96,048
  MACs per step. These are static graph estimates, not GAP8 measurements.
- The native fixed-door environment provides a privileged desktop teacher;
  generic six-DoF environments provide desktop teacher, training, and
  evaluation lanes.
- A generic Python/C mission supervisor now requires target verification before
  navigation and keeps recovery/abort semantics outside learned policy output.
- `flightrl.policy_io_contract` binds raw signal layouts and swappable action
  heads without authorizing hardware deployment.
- The separate desktop [industrial robotics workbench](docs/industry-expansion-20260906.md)
  runs power/production inspection with a drone and wheeled rover, actual WebGPU
  camera observations, shared MuJoCo contacts and a learned visual-servo policy.
- No learned navigation checkpoint in the working repository is current,
  approved, or deployable.
- Generic checkpoint manifests cannot authorize learned live control. A typed,
  byte-bound edge-v3 deployment bundle and the hardware parity/safety gates do
  not exist yet.

The exact model, observation, action, reset, wire, and promotion contracts are
in [docs/edge_navigation_v3.md](docs/edge_navigation_v3.md). Artifact lifecycle
and the first-review cleanup boundary are in
[docs/evidence/README.md](docs/evidence/README.md).

## Development sequence

The shortest valid path is:

1. Develop and verify the environment, privileged teacher, and edge-shaped
   actor on macOS.
2. Register fresh frame-safe camera supervision, then implement the adapter,
   trainer/distillation path, and held-out evaluator for the exact edge-v3
   student.
3. Freeze preprocessing and operators, then establish recurrent-sequence parity
   through PyTorch float, host float C, calibrated int8 C, and GAP8.
4. Prove actual ELF memory use and sustained GAP8 latency.
5. Add the CPX proposal protocol and independent STM32 freshness, clamp, slew,
   estimator, geofence, deadman enforcement, and a trusted arrival signal. The
   policy proposal cannot assert mission completion.
6. Promote through capture, replay, passive shadow, and tethered bounded-axis
   gates before any learned mission flight.

Desktop teacher success is supervision evidence, not a deployable checkpoint.
Physical flight is never an implicit consequence of a simulation or replay
gate.

## Setup and verification

```bash
python -m pip install -e ".[dev]" --no-build-isolation
python setup.py build_ext --inplace --force
python -m pytest -q
```

Optional dependencies are grouped by use:

```bash
python -m pip install -e ".[mujoco]" --no-build-isolation
python -m pip install -e ".[hardware,dev]" --no-build-isolation
```

PufferLib is a separate checkout rather than a FlightRL package dependency.
Pass it with `--puffer-root`/`--pufferlib-root` or use the supported environment
variable for the relevant script.

Useful narrow checks:

```bash
python scripts/smoke_test.py --config configs/tasks/hover.toml
python scripts/benchmark_sixdof_native.py --num-envs 8192 --steps 1000
python scripts/report_edge_budget.py
python -m pytest -q tests/test_puffer4_edge_policy.py
```

The corrected fixed-door privileged-teacher smoke is:

```bash
python scripts/evaluate_puffer_fixed_door_teacher.py \
  --agents 64 \
  --steps 6000 \
  --seed 10011

python scripts/evaluate_puffer_fixed_door_teacher.py \
  --agents 64 \
  --steps 6000 \
  --seed 10012 \
  --obstacle-probability 1
```

The first command is obstacle-free; the second stresses procedurally generated
obstacles whose route and tracking geometry pass the scene-validity contract.
Neither covers arbitrary obstacles, lighting, latency, different room
footprints, learned-student behavior, or physical flight. The results must not
be described as general navigation success.

## Retained research lanes

The repository intentionally keeps three desktop research surfaces:

- the native C/Ocean and Python planar scaffold for small simulator and export
  contract tests only; its legacy learned-policy producer and unversioned
  checkpoint loader are retired;
- native C plus MuJoCo six-DoF environments for teacher, dynamics, reward,
  PPO, imitation, challenge, and parity work;
- the native fixed-door privileged teacher for approach/settle feasibility and
  supervision generation.

They do not define the onboard observation/action ABI. Same-shaped checkpoints
from an older policy or mission contract are rejected rather than partially
loaded or reinterpreted.

Representative desktop commands:

```bash
python scripts/train_sixdof_teacher.py --task multitask
python scripts/evaluate_sixdof_checkpoint.py --teacher \
  --task position_yaw,obstacle_avoidance,circle
python scripts/benchmark_mujoco_sixdof.py --env-counts 1 8 32 128 --steps 300
```

The old fixed-door learned actor/trainer was retired because its observation
and action contracts are not edge-v3. The retained fixed-door path is a
privileged teacher only; the next learned student must use the exact edge-v3
adapter and contract.

## Hardware boundary

The retained hardware tools cover firmware/camera recovery, telemetry capture,
nonlearned bring-up, calibration evidence, and non-actuating grounding. Start
with [docs/crazyflie_bringup.md](docs/crazyflie_bringup.md) and
[docs/ai_deck_camera_setup.md](docs/ai_deck_camera_setup.md).

No current command may use a generic learned checkpoint to actuate the drone.
Do not treat dry-run, replay, desktop export, passive monitor, teacher, or
simulation-gate output as live authority.

## Repository map

- `src/flightrl/native/`: C semantic core, versioned host interface, and Python
  adapter.
- `src/flightrl/artifact_identity.py`: canonical content and file identities.
- `src/flightrl/scenario_bundle.py`: offline compilation of explicit vehicle,
  terrain, sensor, frame, and resolved-mission contracts.
- `src/flightrl/policy_io_contract.py`: hash-bound raw observation and action-head
  layout compiler.
- `src/flightrl/sixdof/`: six-DoF tasks, policies, PPO/imitation, and reports.
- `src/flightrl/mujoco/`: independent rigid-body/contact validation lane.
- `src/flightrl/puffer4_door_*`: fixed-door privileged teacher, mission metric,
  and native export contracts.
- `src/flightrl/puffer4_edge_*`: canonical edge-v3 contract, actor reference,
  and static budget.
- `src/flightrl/hardware/`: capture, telemetry, bring-up, and safety helpers.
- `src/flightrl/sim2real/`: evidence reports and fail-closed authority boundary.
- `scripts/`: explicit train, evaluate, export, replay, and hardware utilities.
- `tests/`: contract, math, parity, safety, and regression coverage.
- `docs/research/README.md`: current research pointers; superseded run logs and
  handoffs live only in Git history or the local cold archive.

## Contribution and license

- [MIT license](LICENSE)
- [Contributing guide](CONTRIBUTING.md)
- [Code of conduct](CODE_OF_CONDUCT.md)
- [Security policy](SECURITY.md)
