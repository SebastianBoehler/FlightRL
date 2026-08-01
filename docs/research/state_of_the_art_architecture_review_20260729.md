# FlightRL state-of-the-art architecture review

Date: 2026-07-29. The [experiment retrospective](flightrl_experiment_retrospective_20260729.md) is the authoritative local handoff. Claims below come from opened papers, repositories, source code, licenses, issues, and local artifacts; promotional throughput and videos are not treated as quantitative evidence.

Inspected source snapshots: PufferLib `c5d3c637`, tensaur/drone `096e91e6`, Emerge-Lab/DroneRacing `18b54ae5`, and Isaac Lab 3.0 beta2 `af1bab4d`. Current FlightRL measurements are taken from the retrospective rather than inferred from the dirty worktree.

## Verdict

**Build the next proof in the native C/Puffer lane, but do not build one monolithic semantic flight policy.** Use three deployment boundaries:

1. A semantic grounder/tracker converts a mission category into compact, time-stamped visual evidence. It is supervised and replaceable; it never commands motors.
2. One small recurrent visuomotor policy owns active search, short-term memory, approach, recovery, and local collision avoidance. Combining these tightly coupled partial-observation behaviors is cheaper and more testable than separate learned planners on a 30 g platform.
3. A deterministic safety envelope converts bounded body-frame velocity/yaw-rate requests into stock Crazyflie firmware setpoints. Firmware stabilization, limits, timeout, estimator checks, and human abort remain outside learning.

The first experiment must be **fixed-category `find and approach a door`**, not multi-target conditioning. The student receives camera frames, recurrent state, odometry/IMU, previous executed action, and mission phase, but no target pose, bearing, vector, semantic map, or detector output derived from simulator truth. Target pose is legal only in reward, the privileged teacher, and asymmetric critic. This isolates whether low-resolution vision plus memory can discover and approach a recognizable category at all.

The credible challenger is a **shared end-to-end recurrent door/monitor/sink policy** with a runtime one-hot target token. A second shared condition with a separately supervised compact grounder tests the main modularity question. Both retain the same external firmware safety layer; an end-to-end policy is never allowed to absorb flight safety.

Do not add a world model, photorealistic training stack, dense metric map, VLM, VIO rewrite, or raw motor authority yet. Evidence supports teacher-to-student DAgger plus asymmetric-critic PPO; it does not show that these additions are required for the next observability proof.

### Implementation status on 2026-07-30

The native fixed-door D1 implementation now has the recommended actor/teacher/
critic split and a Mac-hosted shadow runtime. Its current best single-seed
candidate reaches 34.11% completion, 18.60% collision, and 28.79% outside-FOV
completion, so it fails every live-authority threshold. It is approved only for
monitor-only shadow data collection.

One contract deviation is deliberate scaffolding for the first demo: simulation
currently supplies exact rendered `search/track/approach/recover` phase, while
the Mac runtime derives phase from Grounding DINO detections and evidence age.
The policy receives neither target pose nor target vector. Before claiming a
fully deployable D1 result, retrain with noisy simulated detector evidence that
matches the host contract, or remove the externally supplied phase and let the
recurrent actor infer it.

## FlightRL evidence that constrains the choice

| Lane | Observation/action | Measured result | Architectural implication |
|---|---|---|---|
| Native visual environment | `16x12` rendered vision plus privileged local goal intent; native dynamics/render | About **2.05M environment transitions/s** | Keep it as bulk generator, but remove oracle intent for semantic claims. |
| Native PPO | Same environment; compact policy | **81k learning transitions/s**; **220-292k** max-throughput profile | Report learner throughput separately from environment stepping. |
| `fast16_lowlight_v5` | 33,761 parameters; local goal intent; setpoint lane | **98.59%** held-out obstacle success, **1.33%** collision; masked camera **0/100**; `64x48 gray4` stationary shadow **64.86 FPS**, zero drops | Strong local-control upper bound and camera-dependence control; not semantic navigation. |
| Semantic MuJoCo/Python | `64x48 gray4`, `16x16` memory, proprioception, category; firmware setpoints | About **270 SPS**; best run **38.89%** mission completion, **5.56%** collision; v28 **23.53/5.88**; no checkpoint passed shadow | Current semantics are too slow and weak to justify live authority or more architecture. |
| Direct live control | Raw/high-authority control | High-speed drift/crash plus pitch-sign mismatch | Near-term authority must be bounded setpoints/residuals over stock stabilization. |

## What the fixed-door baseline proves

- Randomize room grammar/family, start pose, door instance, dimensions, frame/panel geometry, material, lighting, distractors, occlusion, and target initially inside/outside the FOV. Hold out factor combinations, geometry ranges, and whole room grammars.
- Success proves category-specific visual search, recurrent out-of-FOV memory, approach behavior, and obstacle-aware control without an oracle target vector. Camera masking and randomized odometry frames test leakage.
- It does **not** prove runtime semantic conditioning, language grounding, counterfactual target choice, open-vocabulary generalization, or transfer to monitor/sink. A policy may simply learn a specialized door appearance prior.
- Per-target checkpoints are a useful demo/product lane, regression oracle, and source of teachers. They trade an easy deployment contract for `O(number of targets)` training and maintenance, and cannot support a claim that one policy follows semantic missions.
- **Semantic supervision** means masks, visibility, target pose, centroid, phase, teacher actions, or privileged critic state are used only to form losses/reward during training. **Semantic mission input** is a runtime command that must change which object the same policy selects in the same scene. Only token-swap counterfactual evaluation proves the latter.

## Recommended policy contract

- Camera: fixed `uint8` monochrome `64x48`, 15-30 Hz, fixed exposure metadata plus `frame_valid` and `frame_age`; temporal information is carried by recurrence rather than an expanding frame stack. A supervised pretest must show door/monitor/sink detectability at this resolution.
- Nonvisual student state: body velocity (3), gyro (3), gravity/body-z (3), altitude/range (1), bounded odometry delta and yaw delta (4), previous **executed** body setpoint (4), mission phase one-hot (`search`, `track`, `approach`, `recover`), and validity/age bits. Phase must be computed from student-visible evidence, timers, and safety state, never simulator truth. No global room pose or target-relative quantity.
- Mission: no target input for fixed-door; later `uint8 target_id`/three-way one-hot only. Any grounder output is an explicit compact contract such as visible/confidence, normalized centroid/scale, and evidence age, never target pose.
- Actor: fixed-shape convolution stem plus at most 64 recurrent units and linear head; target at most 50k parameters, at most 64 KiB int8 weights, bounded activations, no attention/dynamic allocation. Auxiliary heads are removed at export.
- Action at 20 Hz: normalized body-frame `vx, vy, vz, yaw_rate`; deterministic clamps, slew limits, deadman, altitude/geofence checks, and estimator validity precede the stock firmware controller. No direct motor/RPM action.
- Export: first require bit-exact float C parity using PufferNet-style kernels; then quantization-aware int8 parity using DORY/CMSIS-NN-compatible conv/dense operations and a separately verified small recurrent kernel. Measure capture, preprocessing, inference, link, and controller latency independently.

## Learning allocation

- Use **supervised learning** for category visibility, centroid/scale, masks or keypoints, evidence confidence/age, and optional motion/time-to-collision/egomotion auxiliary heads. Simulator target pose may label these heads but is never a deployed input.
- Use a **privileged state teacher and DAgger** to cover early failures, recovery, occlusion, and outside-FOV search. Aggregate labels on student-visited states rather than relying on offline behavioral cloning.
- Use **asymmetric-critic PPO** for active yaw/search, long-horizon exploration, approach speed, obstacle tradeoffs, and recovery after imitation. The actor remains restricted to the deployment observation contract.
- Treat body yaw as the first active-vision mechanism. Do not add a learned fovea or ROI controller until the resolution pretest shows that a fixed full frame is insufficient; a detector-triggered fixed crop is cheaper and easier to export.
- Add a **world model** only as a later measured challenger for occupancy/visibility memory, time-to-collision, semantic persistence, or residual dynamics. It is not a room generator.

## Evidence table: native, Puffer, and nano-UAV systems

Every row records task; observation/semantics; action/control; stack/model/memory/training; throughput/hardware/sim-to-real; code/data/license/reproducibility; and FlightRL value.

| System | Task and observation / semantic input | Action and control | Stack, model, memory, training | Throughput, hardware, sim-to-real | Code, data, license, reproducibility | FlightRL reuse |
|---|---|---|---|---|---|---|
| [PufferLib `ocean/drone`](https://github.com/PufferAI/PufferLib/tree/master/ocean/drone) | Hover/race; 21 privileged values: body velocity, angular rate, quaternion, coarse/fine target vectors, target normal, task one-hot; no camera/language | Four normalized motor/RPM commands, 100 Hz; physics 500 Hz | Native C/RK4; PufferNet 21 -> 64x2 -> 4; PPO; no policy recurrence in current environment | [Generic docs](https://puffer.ai/docs.html) claim up to 20M CUDA and 5M Torch learning SPS, but give no drone-specific environment/learner split or quantitative hardware trials | MIT; current code and operation parity tests; flat fp32 `weights.bin` | Directly reuse trainer/vectorization/serialization and float kernels; port dynamics tests only, not observation/action contract. |
| [tensaur/drone](https://github.com/tensaur/drone) / firmware | Hover/race; simulator 19 privileged values; firmware network takes 23 inputs (19 plus four zeros); no camera/language | Direct motors; learned policy 100 Hz, motor update 1 kHz, PID fallback | Firmware 23 -> 64 -> 16 -> LSTM16 -> actor/value; 4,841 fp32 weights (19,364 B); PPO, 2,048 agents, 64 drones/env, 46.927M steps in config | Public videos/code show flight, but no controlled trial count, failure rate, or drone-specific SPS | Outer MIT; Crazyflie firmware submodule GPL; source at inspected commit `096e91e`; reproducibility weakened by 19/23 mismatch | Surgically port dynamics/randomization and firmware integration ideas; do not reuse direct motor authority or network contract. |
| [Emerge-Lab/DroneRacing](https://github.com/Emerge-Lab/DroneRacing) | Privileged-state race/swarm; raylib is human rendering, not policy vision; no semantics | Direct low-level actuation | Puffer fork/native environment; policy path is incomplete/TODO in inspected source | No reproducible benchmark or sim-to-real result | MIT; two-star classroom fork, placeholder README commands and minimal tests | Conceptual race/task scaffolding only. |
| [PULP-DroNet v1/v2](https://github.com/pulp-platform/pulp-dronet), [paper](https://arxiv.org/abs/1905.04166) | `200x200` gray; steering regression plus collision probability; no mission semantics | Forward speed/yaw steering above firmware stabilization | Shallow residual CNN, about 320k parameters/1.3 MB fp32; supervised driving/drone data; feedforward | v1 6-18 FPS, 64-284 mW; v2 int8 9-17 FPS, 35-102 mW on GAP8; 27 g class flight, 113 m unseen route and dynamic avoidance reported | Apache-2.0 code; NEMO/DORY/GAPflow paths; datasets vary; current Bitcraze warning says GreenWaves AutoTiler unavailable | Direct deployment and bounded-command reference; reuse DORY pipeline/dual-head formulation, not its nonsemantic road prior. |
| [Tiny-PULP-DroNet v3](https://arxiv.org/abs/2407.12675) | Same steering/collision task; 66k unified onboard images; no mission input | Firmware-level navigation commands | Distilled feedforward int8 CNN family: 51k/17k/6.6k/2.9k parameters; supervised distillation | 34/61/101/139 FPS on GAP8; smallest model 100% on one unseen corridor/U-turn course at 0.5 m/s | Apache-2.0 code/models; dataset CC BY-NC-SA 4.0, a product restriction; field protocol finite | Direct size/rate target and QAT/DORY reference; rerun on FlightRL data and license-clean assets. |
| [Local + global nano-UAV perception](https://arxiv.org/abs/2403.11661) | PULP-DroNet image output plus `8x8` front ToF at 15 Hz; no mission semantics | LUT local avoidance plus global steering, firmware stabilized | Explicit modular fusion; feedforward global CNN | 15/15 successful combined flights; global-only fails static obstacle, local-only fails 90-degree turn | Paper/video; implementation provenance linked to PULP stack, no broad benchmark | Strong evidence to keep semantic/global evidence and last-metre safety separable. Extra ToF hardware is not assumed. |
| [NanoFlowNet](https://arxiv.org/abs/2209.06918) | Paired QQVGA gray frames -> `40x28` dense flow; no semantics | One-byte yaw request over UART; firmware stabilization and vertical oscillation | Tiny CNN, feedforward supervised optical flow | 5.5-9.3 FPS GAP8; 34 g Crazyflie, 0.2 m/s field flight | Paper and field demonstration; no public implementation/data license was verified | Conceptual optional motion auxiliary/flow baseline; dense flow is neither target grounding nor memory. |
| [Edge-FS](https://arxiv.org/abs/1612.06702) | Stereo feature histograms -> optical flow, disparity, velocity/depth; no semantics | Classical velocity control and obstacle avoidance | Hand-designed algorithm, no learned policy or recurrence | 20 Hz on 168 MHz STM32F4 with 192 KiB and 4 g stereo camera; autonomous 40 g pocket drone | Paper and hardware experiments; no reusable licensed implementation verified | Strong classical compute baseline, but requires stereo hardware absent from the AI Deck contract. |
| [LEVIO](https://github.com/ETH-PBL/levio) | `160x120` mono + IMU -> 6-DoF VIO; no semantics | State estimator only | ORB/BRIEF, RANSAC, IMU preintegration, 8-keyframe optimization; classical memory | 20 FPS, <100 mW, <110 KiB L1 on GAP9; EuRoC only, no flight validation | MIT, four commits, C + Python; as-is/no active development | Future GAP9 surgical reference. It does not fit the AI Deck GAP8's 512 KiB L2 unchanged and is not needed before fixed-door. |
| [AI Deck / GAP8 examples](https://github.com/bitcraze/aideck-gap8-examples), [hardware](https://www.bitcraze.io/products/ai-deck/) | HM01B0 mono up to `320x320`; GAP8 8+1 cores, 512 KiB L2, 64 KiB L1; no standard semantic interface | CPX/UART to STM32 firmware | Examples include camera streaming and MobileNet; example licenses must be checked per file | AI Deck adds 4.4 g; local FlightRL QVGA shadow already measured 64.86 FPS with zero drops | Bitcraze repos/source available; MobileNet demo explicitly warns about poor generalization; AutoTiler access is currently broken | Directly reuse capture/CPX/firmware setpoint interfaces; do not treat demos as deployable perception. |

## Evidence table: visual control, racing, and semantic navigation

| System | Task and observation / semantic input | Action and control | Stack, model, memory, training | Throughput, hardware, sim-to-real | Code/data/license/repro | FlightRL value |
|---|---|---|---|---|---|---|
| [Bootstrapping RL with imitation](https://arxiv.org/abs/2403.12203) | Known-track racing; RGB or gate corners; teacher sees full state/next gate; no language | Collective thrust + body rates over low-level controller | PPO privileged teacher -> DAgger visual student -> PPO with privileged critic; ResNet50 128-D + TCN history 32 | Fixed 10M samples; five seeds/100 evals; scratch RL and BC 0%, DAgger about 52-64%, adaptive 72-85%; real Split-S | Paper/project videos; no public training code found | Primary training schedule, but replace ResNet/TCN and race action authority with tiny recurrent setpoint actor. |
| [Swift](https://www.nature.com/articles/s41586-023-06419-4) | Known mapped gates; 30 Hz RGB + IMU -> VIO, gate corners, Kalman state; no language | Collective thrust + body rates, 40 ms sensorimotor latency | Detector/VIO/resection/Kalman -> two-layer MLP; model-free on-policy RL with real residual perception/dynamics models | Champion-level real racing, hundreds of laps; larger racing drone/onboard compute, no nano-UAV claim | Open paper and extensive real evaluation; full integrated stack is not a drop-in repository | Evidence for modular perception/state/control and real residual validation, not dense VIO as a prerequisite. |
| [Deep Drone Acrobatics](https://github.com/uzh-rpg/deep_drone_acrobatics) | Fixed aggressive maneuvers; onboard visual abstraction and inertial history; privileged optimal-controller teacher; no semantics | Low-level acrobatic control over the vehicle controller | Simulation expert demonstrations and iterative DAgger; temporal student policy | At least 15 Hz required; zero-shot real Power Loop/Barrel Roll/Matty Flip, up to 3 g on a larger drone | MIT code/checkpoint; 23 commits, Ubuntu 18.04/ROS Melodic/Python 3.6 | Reproducible evidence for privileged teacher, DAgger, history, and task-specific visual abstraction; not exploration. |
| [Agile Autonomy](https://github.com/uzh-rpg/agile_autonomy) | Unknown clutter; depth + noisy state; no semantics | Predicts short collision-free trajectory consumed by controller | Privileged planner expert -> CNN imitation; Flightmare | Zero-shot real flights on larger platform; inference must exceed 15 Hz | GPL-3.0, pretrained model and old ROS/TF2.4 setup | Conceptual endpoint separation and teacher data; license/dependencies preclude direct port. |
| [SOUS VIDE](https://arxiv.org/abs/2412.16346) | Scene-specific RGB, optical flow, IMU; no mission language | Collective thrust/body rates at 20 Hz | Gaussian-splat rendering + MPC labels (100-300k), SV-Net and runtime dynamics adaptation | Rendering up to 130 FPS; 105 hardware experiments on larger drone | Paper/project code/data; scene reconstruction required | Photoreal independent challenger for appearance transfer, not a procedural unseen-room generator. |
| [SINGER](https://arxiv.org/abs/2509.18610) | Language target, monocular RGB, IMU/magnetometer/flow/range; CLIPSeg heatmap from semantic 3DGS/RRT* expert | Collective thrust/body rates | About 700k-1M imitation pairs; CLIPSeg ViT-B/16 about 86M, 12 Hz on Orin Nano; policy 20 Hz | 6x5 hardware trials; target starts visible; reported out-of-FOV/collision failures | Paper/project; no reusable licensed implementation found | Conceptual grounding/expert split and failure cases; compute and initial-visibility assumptions do not transfer. |
| [SemExp](https://github.com/devendrachaplot/Object-Goal-Navigation) | Object category + RGB-D + pose -> semantic map | Discrete local navigation through deterministic planner | Supervised Mask R-CNN/map; PPO global-goal policy; explicit long-term memory | Pretrained Gibson: 0.657 success, 0.339 SPL; ground robot simulation | MIT, pretrained; one commit, Habitat 0.1.5/PyTorch 1.6 | Transfer decomposition and metrics only; dense RGB-D map/stack is too heavy and stale. |
| [VLFM](https://github.com/rai-opensource/vlfm) | Language/object target + RGB-D; occupancy frontiers and VLM value map | Local PointNav policy | Frozen VLM/detector + explicit maps/frontiers; no end-to-end training | Habitat and Spot real office demonstrations; GPU/depth robot | MIT, active code/tests, heavy GroundingDINO/LAVIS/YOLO dependencies | Conceptual offboard/open-vocabulary grounder and exploration scoring; not onboard Crazyflie code. |
| [GOAT](https://arxiv.org/abs/2311.06430) | Sequential category/language/image goals + RGB-D/GPS; instance memory | Ground-robot navigation | Instance-aware modular memory and exploration | 90 h, 9 homes, 675 goals, 200+ instances; 83% reported success | Paper/benchmark assets; repository licensing is unclear | Reuse counterfactual/sequential benchmark ideas, not mapping stack or performance numbers. |

The transferable ObjectNav/VLN lesson is not "install Habitat on the drone." It is to preserve explicit contracts among grounding, evidence aging/tracking, exploration memory, local avoidance, and control. On the Crazyflie, dense metric RGB-D maps and onboard VLMs become compact recurrent/topological evidence; firmware safety stays deterministic.

## Evidence table: simulators and procedural generation

Simulator rows are infrastructure, not complete agents. Observation, action, model size, recurrence, and training method are task-defined and therefore not attributed to a simulator unless an official bundled task is explicitly described.

| Stack | Physics/render/semantics and vectorization | Reported speed vs learner speed | Platform/license/reproducibility | FlightRL role |
|---|---|---|---|
| FlightRL native C/Puffer | Native six-DoF plus current ray/primitive vision; semantics/procedural rooms are controllable; fused vector stepping | **2.05M env SPS**, **81k PPO learning SPS**, 220-292k max profile | Mac/local, repository-native | Primary bulk training and causal ablation lane. |
| [MuJoCo](https://github.com/google-deepmind/mujoco) | High-quality contact/rigid dynamics, C API/OpenGL; semantics need custom assets/IDs; vectorization external | Current semantic lineage about **270 SPS**, not a MuJoCo engine limit | Apache-2.0, Mac/local | Independent dynamics/contact and Python reference validation, not bulk semantic training. |
| [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) | C++/Magnum + Bullet; scans/CAD, RGB-D/semantic sensors, scene datasets | Official: several-thousand single-thread render FPS, >10k multiprocess; Fetch RGB-D physics >8k SPS; no learner number | MIT; attached-display Mac possible, headless EGL not Mac; Meta maintenance stops after 0.3.4 | Rich semantic/appearance validation and dataset benchmark only. |
| [Isaac Lab 3.0 beta2](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/) | GPU PhysX/RTX, USD semantics, massive vectorization; bundled Bitcraze `cf2x.usd` | Showroom gives no training SPS. `quadcopter.py` merely applies `mass*g/4` to prop bodies | BSD-3 Lab code plus Isaac Sim terms; NVIDIA Linux/Windows, not Mac | Independent Crazyflie geometry/PhysX and richer visual validator. The bundled direct task is 12-D privileged hover, total thrust + body moments, not camera autonomy or firmware equivalence. |
| [Aerial Gym](https://github.com/ntnu-arl/aerial_gym_simulator) | Isaac Gym/PhysX GPU drones, Warp depth/segmentation, thousands of environments, controllers | README claims state tasks <1 min and vision <1 h without a sufficient hardware-normalized env/learner split | BSD-3; legacy Isaac Gym Preview 4, Ubuntu/CUDA | Conceptual GPU sensor/control patterns; high integration cost and no Mac path. |
| [Flightmare](https://github.com/uzh-rpg/flightmare) | Decoupled C++ dynamics + Unity RGB/depth/segmentation; hundreds of drones | Paper reports up to 200k dynamics SPS on six cores/150 drones and about 230 FPS for five cameras; learner figures are setup-specific | MIT, old Unity/ZeroMQ/Python/ROS stack | Surgical richer renderer/validation only; not worth replacing native core. |
| [AirSim](https://github.com/microsoft/AirSim) | Unreal/PhysX, RGB/depth/segmentation, PX4 SITL/HIL | No comparable high-throughput semantic learner result | MIT but upstream discontinued/last release 2022; heavy integration | Conceptual SITL/HIL and visuals only. |
| [gym-pybullet-drones](https://github.com/learnsyslab/gym-pybullet-drones) | PyBullet Crazyflie dynamics, cameras, Gymnasium; Betaflight/Crazyflie firmware SITL | No relevant published visual learner throughput | MIT, current, Apple Silicon tested | Directly reuse firmware/controller-contract validation and regression scenarios, not bulk rendering. |
| [RotorS](https://github.com/ethz-asl/rotors_simulator) | Gazebo/ROS rotor dynamics and sensors; no native vectorized learner | No relevant throughput | MIT, targets Ubuntu 16.04/ROS Kinetic era | Historical dynamics/controller reference only. |
| [Crazyflow](https://github.com/utiasDSL/crazyflow) | JAX/MJX differentiable, batched Crazyflie fitted dynamics/controllers; no semantic camera | Reports 3.3M CPU physics SPS (64 worlds), up to 914M GPU SPS (262k worlds); physics only | MIT, modern code | Surgically port identified parameters/controller tests; never compare its physics-only number to rendered PPO. |
| [DiffAero](https://github.com/flyingbitac/diffaero) | PyTorch GPU aerial dynamics, depth/LiDAR and differentiable PPO/Dreamer/BPTT/SHAC; no semantic RGB | MAD reports 121k env interactions/s including map construction, not policy learning | BSD-3, CUDA-centric | World-model/geometry challenger platform, not Mac/native integration target. |

No renderer is a universal winner. Native C is sufficient for the next **architecture/observability** proof if it supports held-out geometry, materials, illumination, occlusion, distractors, and outside-FOV starts. It is insufficient evidence for appearance sim-to-real. Train in native C; validate independently on recorded AI Deck frames plus MuJoCo and one richer renderer/asset lane. Add photorealistic training only if a supervised real-versus-native detectability test or policy replay identifies an appearance gap that domain randomization cannot close.

## Evidence table: world models

| System | Observation/action/model/training | Throughput/hardware/sim-to-real | Code/license/repro | What it actually improves; FlightRL decision |
|---|---|---|---|
| [DreamerV3](https://github.com/danijar/dreamerv3) | Generic recurrent state-space model; actor-critic trained on imagined latent trajectories | Domain dependent; large GPU learner, no aerial deployment claim | MIT JAX reimplementation, Mac/Linux, configs/results | Sample efficiency/representation/control framework only. It does not supply geometry not present in observations/data. |
| [DayDreamer](https://github.com/danijar/daydreamer) | Real robot replay; fused visual/proprio latent model and imagined actor-critic | Physical quadruped/arm/wheeled tasks; learner and actor are separate processes, usually GPU-backed | Official TF2 repo has one commit and no visible top-level license | Shows real-data adaptation/control, not nano-UAV inference or room generation. |
| [Dream to Fly](https://arxiv.org/abs/2501.14377) | Raw `64x64` RGB -> collective thrust/body rates; DreamerV3 Large (4x768, RSSM 2048); known race tracks | 10-20M interactions, up to about 240 h on RTX8000; HIL uses rendered images with motion capture, not onboard real-image transfer | Paper/project; five seeds, no compact deployment artifact | Improves pixel-control sample efficiency and gaze on known racing geometry; too large for next FlightRL step. |
| [SkyDreamer](https://arxiv.org/abs/2510.14783) | GateNet binary mask + rates/RPM -> four motors; informed Dreamer decodes privileged state/dynamics; recurrent | Real onboard Orin NX/TensorRT, up to 21 m/s and 6 g on mapped gates | Paper; no reusable code/license found | Improves racing state/parameter estimation and control from a task-specific mask; not semantic search or unseen rooms. |
| [Dreaming Falcon](https://arxiv.org/abs/2511.18243) | Learned force/moment or unconstrained RNN quadrotor dynamics for model-based RL | Both fit train data; reported OOD rollout divergence and policy convergence failures | Paper, no mature reuse path | Negative evidence: a learned dynamics model can worsen control outside its data. |
| [MAD](https://arxiv.org/abs/2606.04534) | `18x32` depth + goal-directed velocity/proprio -> acceleration; recurrent latent reconstructs 4,000-cell occupancy/visibility; Dreamer/PPO/SHAC | 121k env interactions/s including maps; RealSense D435i/VIO larger drone, 5.05 m/s real forest flight | Paper/DiffAero; code path emerging | Improves observed-geometry memory, collision control, and transfer. It reconstructs visibility/occupancy; it does not invent reliable unseen geometry. Later challenger if simple recurrence fails. |
| [AirDreamer](https://arxiv.org/abs/2606.03252) | `48x80` depth + 15-D state including oracle goal direction/distance -> body velocity/yaw setpoint; DreamerV3, 101.7M train/57.9M deploy params | 2.5M steps, 38 h 43 min on 2x RTX4090D; real larger drone up to 1.8 m/s | Paper says code will be public; no current repository | Improves detours/active yaw by 5.3 points over best baseline, but uses oracle goal and is orders too large. |

A world model predicts a conditional distribution from observation/action history and its training distribution. It cannot be assumed to reveal an unobserved door or generate accurate unseen room geometry. Introduce one only after the recurrent baseline, and only with a measured target such as occupancy/visibility memory, time-to-collision, latent semantic persistence, or residual dynamics.

## Fixed multi-seed experiment

Use training seeds **`{11, 23, 47, 71, 101}`**, identical procedural streams by episode index, **50M environment transitions per seed/condition**, identical action limits, and at most 1.25x parameter variation. For imitation conditions, the first 5M transitions are on-policy DAgger collection and the remaining 45M are asymmetric-critic PPO; no post-hoc budget extension. Evaluate each seed on at least 1,000 deterministic held-out episodes, stratified by room grammar, target geometry, lighting, occlusion, distractor, and initial visibility.

| ID | Student information | Training | Question |
|---|---|---|---|
| `A0-oracle` | Current camera/proprioception plus privileged local target intent | PPO | Local-control upper bound; not semantic. |
| `D0-ff` | Fixed door, no target input, no recurrence | DAgger + PPO | Is temporal memory actually needed? |
| `D1-door` | Fixed door contract above, recurrent | DAgger + PPO | Primary no-oracle baseline. |
| `D2-scratch` | Same as `D1-door` | PPO from scratch | Does the racing-derived teacher/DAgger stack earn its complexity? |
| `E3-shared` | Raw frames + door/monitor/sink token, recurrent | DAgger + PPO | End-to-end target-conditioned challenger. |
| `M3-modular` | Token + compact output from separately supervised grounder/tracker + recurrent navigator | Supervised grounder, then DAgger + PPO | Does explicit grounding improve multi-target reliability? |

Pre-register thresholds on pooled episodes with seed-level bootstrap and episode-level Wilson 95% intervals:

- `A0-oracle`: at least 95% completion and at most 2% collision, otherwise the environment/control upper bound is invalid.
- `D1-door`: at least 80% completion, at most 3% collision, at least 70% completion when initially outside FOV, at least 60% in every pre-registered factor group, and masked-camera completion at most 5%.
- `E3/M3`: at least 70% aggregate completion, at most 4% collision, at least 60% for every target, door within 10 points of `D1`, and at least 80% target compliance under same-scene mission-token swaps.
- Choose modular `M3` if its paired lower confidence bound exceeds `E3` by 5 completion points or 3 worst-group points. Choose simpler `E3` only if it is within 3 points on aggregate and worst-target success, is no worse on collision, and passes token swaps.

Kill criteria: camera-mask success >5% or odometry-frame randomization collapse indicates leakage; outside-FOV `D1` <70% kills multi-target work; recurrence gain <3 points drops recurrence; DAgger gain <5 points drops teacher complexity; token compliance <80% keeps per-target checkpoints and kills the shared-policy claim; collision >5% kills any live-readiness work. A later world-model challenger must add at least 5 completion points or 10 outside-FOV points at equal observations while meeting the deployment budget.

## Defensible scaling and validation

"50 rooms is enough" has no support. Treat a room as a sampled factorization, not a dataset item. For the surviving architecture, train with **8, 32, 128, and 512 base topology seeds**, while continuously resampling object geometry/material/light/start/occlusion/distractors. Test on disjoint topology grammars and factor ranges. Fit `error(N) = a*N^-b + c` across five seeds; claim saturation only if the 128-to-512 change is <2 points with a 95% interval, all worst groups pass, and independent visual validation does not regress. Non-monotonic scaling means repair the generator/task, not add rooms blindly.

Independent validation is intentionally out of distribution: (1) held-out native grammar; (2) MuJoCo dynamics/contact replay; (3) recorded AI Deck frame sequences with frame timing/dropouts; (4) one richer visual lane, initially Isaac's exact Crazyflie USD or Habitat/Flightmare assets. No validation scene contributes training images, teacher labels, normalization statistics, or stopping decisions.

## Repository reuse and port plan

- **Reuse directly:** PufferLib vector/trainer interfaces, flat weight serialization, float C kernel parity tests; Bitcraze camera/CPX/setpoint/deadman interfaces; PULP-DroNet v3's license-compatible model/deployment code; existing FlightRL native dynamics/render and measured camera pipeline.
- **Port surgically:** tensaur/Puffer drone dynamics and randomization tests after reconciling units; Crazyflow identified parameters/controller tests; gym-pybullet-drones firmware SITL scenarios; Tiny-PULP QAT/DORY operations; Isaac's Crazyflie geometry/PhysX reference; LEVIO frontend ideas only for a future GAP9 lane.
- **Conceptual reference only:** Emerge DroneRacing, Swift, Bootstrapping RL+IL, Deep Drone Acrobatics, Agile Autonomy, NanoFlowNet/Edge-FS, SOUS VIDE, SINGER, SemExp, VLFM, GOAT, FlightBench, Aerial Gym, Flightmare/AirSim/RotorS integration stacks, and all current aerial world models.
- **License boundaries:** Crazyflie firmware and FlightBench/Agile Autonomy GPL code must remain separated from permissive native code; PULP v3 data is noncommercial CC BY-NC-SA; GOAT/DayDreamer/SINGER artifacts without clear reuse licenses cannot enter product code or training data.

## Smallest discriminating experiment and stages

1. **Observability pretest:** first train a tiny supervised visibility/centroid/scale head on native `64x48` for doors only; test held-out native plus labeled real AI Deck recordings. Stop if door visibility AUROC <0.90 or centroid median error >0.12 normalized image width; improve rendering/resolution before RL. Monitor/sink observability is tested only after the fixed-door gate passes.
2. **Architecture screen:** pre-register the fixed six-condition, five-seed matrix, but run `A0/D0/D1/D2` first. Run `E3/M3` only if `D1` passes its fixed thresholds. This distinguishes oracle local control, fixed-category search, recurrence, teacher/DAgger value, shared end-to-end conditioning, and modular grounding without starting multi-target work prematurely.
3. **Scaling:** run only `D1` and the winning shared architecture over the 8/32/128/512 topology curve. Do not add a world model during this stage.
4. **Independent simulation:** replay deterministic scenario manifests in MuJoCo and the richer visual lane; require at least 70% completion, at most 5% collision, and no action-axis/sign mismatch. Measure environment SPS and learner SPS separately.
5. **Deployment:** freeze the observation/action schema; float-C parity, then int8 parity; require max action error <=1% of range, worst-case inference <=20 ms, no dynamic allocation, and end-to-end frame-to-command p99 <=50 ms. Run firmware SITL/controller clamps before hardware.
6. **Hardware shadow:** stationary and firmware-held-hover shadow only; camera masks, stale frames, dropouts, lighting, and out-of-FOV door sweeps; zero learned authority. Require zero dropped-control deadlines, zero nonfinite outputs, and 100% deadman/clamp tests.
7. **Next bounded live gate, not executed here:** firmware-held hover with learned **yaw only**, translation hard-zero, <=8 deg/s, <=15 s, open protected area, human abort, charged battery, readiness artifact bound to checkpoint/hash/contract. Approach authority is a later gate only after yaw search and replay pass. Direct motors/raw attitude remain prohibited.

## Unsupported or stale current notes

- `docs/native_visual_training_fast_lane.md:190` calls tensaur's path a "generated C PufferNet forward pass." It is generic hand-written fp32 kernels plus generated weight data, not per-model C generation or int8 deployment; simulator 19 versus firmware 23 inputs also requires resolution.
- `docs/native_visual_training_fast_lane.md:205` makes semantic target IDs the immediate next native task. The no-target fixed-door baseline and observability test must precede this.
- `docs/research/semantic_mission_architecture_20260726.md:487` calls a checkpoint "accepted," and lines 526/553/576 nominate live gates. The later authoritative retrospective says **no semantic checkpoint passed the shadow gate**; these approvals are stale.
- `docs/research/pulp_dronet_2019_analysis.md:13` says obstacle tasks, six-DoF dynamics, sensor models, and camera paths are missing. Those components now exist; its bounded-perception/firmware split remains valid.
- Old comparisons of "Puffer 21 observations versus FlightRL 28" are stale: current Puffer is 21, current tensaur simulation is 19, its firmware network is 23, and current FlightRL visual contracts are image plus state. Contract-level comparisons must name commit and task.
- No opened Puffer/tensaur source supports a quantitative "15 seconds to train," drone-specific SPS, crash rate, or statistically controlled sim-to-real claim. Generic Puffer throughput and public flight videos cannot fill those fields.
- AutoTiler availability/readiness claims are stale: [Bitcraze currently warns](https://www.bitcraze.io/documentation/tutorials/getting-started-with-aideck/) that GreenWaves download access is unavailable; DORY is the reproducible open route.
- Any claim that a world model generates correct unseen rooms, that an arbitrary room count is sufficient, or that `16x12` is semantically adequate is unsupported until the pre-registered tests above pass.

## Bottom line

Puffer/native C remains strategically justified because FlightRL has measured rendered environment throughput, compact policies, deterministic export work, and a real embedded destination. The evidence does **not** justify forcing physics, semantics, perception, memory, and safety into one Puffer policy. The next result should be a narrow causal claim: a tiny recurrent camera policy can or cannot find and approach a randomized door without oracle goal intent. Only after that passes should FlightRL pay for shared semantic conditioning, and only the fixed token-swap comparison can show that semantic mission input adds behavior beyond semantic supervision.
