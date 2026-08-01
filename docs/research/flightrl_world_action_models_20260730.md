# World Action Models: Research and FlightRL Feasibility

**Date:** 2026-07-30  
**Scope:** Recurrent PufferLib visual door search on a firmware-stabilized Crazyflie  
**Decision:** Do not add a World Action Model (WAM) to the overnight or first-live-demo path.

## Bottom line

Current WAM research is relevant as a longer-term representation-learning direction, but it does
not remove FlightRL's immediate bottlenecks: action-contract alignment, yaw-scale transfer,
outside-FOV recovery, and behavior-cloning distribution shift. Published WAM systems are also
several orders of magnitude too large for the AI-deck lane and generally too slow for the
approximately 65 Hz camera/control path.

After the overnight comparisons, the defensible sequence is:

1. retain T61 pure BC as the strongest simulation bundle; the matched seed-11
   DAgger treatment was worse and was stopped;
2. preserve the recurrent observation, previous-action, and unit-bearing action
   contracts;
3. collect and review the non-actuating v59 reference shadow, then optionally a
   separately named T61 shadow;
4. resolve the failed 8 deg/s simulation gate before any learned authority;
5. consider a small training-only action-conditioned latent auxiliary only
   after a measured representation or predictive-memory failure.

No WAM result changes tonight's safety boundary or morning gate. It does not justify physical
drone operation, radio commands, live authority, or replacing firmware stabilization.

## What “World Action Model” means

### Verified paper facts

There is no single canonical architecture named “the WAM.” The 2026
[WAM survey](https://arxiv.org/abs/2605.12090) uses the term for models that jointly learn
future-state prediction and action generation. Individual papers use different mechanisms:
video diffusion, latent dynamics, inverse dynamics, autoregressive action generation, or a
training-only predictive head.

This is distinct from:

- an **action-conditioned world model**, which predicts what follows a supplied action and may
  still require a separate planner or policy;
- a **vision-language-action policy**, which maps observations and instructions directly to
  actions without necessarily predicting a future;
- an **inverse-dynamics model**, which infers an action between two states and is often only one
  component of a WAM;
- latent model-based RL such as
  [DreamerV3](https://www.nature.com/articles/s41586-025-08744-2), which learns a world model for
  imagined rollouts but is not the same recent video-policy family.

“WHAM” can also mean Microsoft's separate gameplay
[World and Human Action Model](https://www.nature.com/articles/s41586-025-08600-3).

### Screenshot attribution caveat

The screenshot depicts an instruction, egocentric image, articulated pose/action, predicted next
image, and next action. That is a generic WAM explanation and resembles whole-body systems such as
[MotionWAM](https://arxiv.org/abs/2606.09215), but the image alone does not identify a specific
paper or implementation. It should not be treated as evidence for a reproducible model.

## Ranked evidence map for FlightRL

Rank is by relevance to semantic aerial navigation, not by benchmark score.

| Rank | Work | Verified result | FlightRL relevance and limit |
|---:|---|---|---|
| 1 | [WorldFly](https://arxiv.org/abs/2606.06147) | UAV-specific WAM trained on more than 4,000 AirSim trajectories. It reports 87% on TEST-EASY but 31% at unseen TEST-HARD intersections. Its primitives include 3/6/9 m translations and ±30° turns. | Closest domain match, but simulation-only and its coarse action space does not match bounded Crazyflie yaw/translation. The unseen-layout drop argues against assuming that video prediction solves semantic search. |
| 2 | [NavWAM](https://arxiv.org/abs/2606.13494) | A 2B-parameter ground-navigation model. In a matched ablation, future-image supervision improved horizon-8 ATE from 0.262 to 0.192 and RPE from 0.103 to 0.070. | Best causal evidence that future prediction can regularize an action policy. It supports testing an auxiliary predictive target, not importing its full model. |
| 3 | [EgoWAM](https://arxiv.org/abs/2607.08436) | Its pixel, DINO-feature, or stabilized-3D-flow world head is used during training and removed at inference. DINO features were strongest out of distribution; raw-pixel prediction was weaker. | Most applicable design pattern: improve the recurrent representation without adding deployment latency. Evidence is from manipulation, not flight. |
| 4 | [DynaMo](https://openreview.net/forum?id=vUrOuc6NR3) | NeurIPS 2024 work pretrains forward and inverse latent dynamics, then transfers the representation to behavior cloning. | A compact precursor that is closer to FlightRL's data scale and actor than video diffusion, but still needs a measured representation failure to justify it. |
| 5 | [MotionWAM](https://arxiv.org/abs/2606.09215) | A 2.5B-parameter humanoid WAM trained with 2,136 hours of video plus task demonstrations. Reported failures include objects leaving the field of view and camera-viewpoint drift. | Demonstrates that WAMs do not inherently solve the exact outside-FOV problem currently limiting FlightRL. Embodiment and compute are mismatched. |
| 6 | [DreamZero](https://arxiv.org/abs/2602.15922) | A 14B-parameter video-diffusion WAM pretrained on roughly 500 hours of robot data; optimized inference is reported at about 7 Hz on GB200-class hardware. | Strong transfer result on top of enormous pretraining, but unusable for the current edge or low-latency lane. |
| 7 | [Cosmos Policy](https://arxiv.org/abs/2601.16163) | A 2B-parameter video-policy approach; its [official repository](https://github.com/nvlabs/cosmos-policy) describes H100-class training and multi-gigabyte inference requirements. | Useful evidence that planning with a pretrained video model is possible, but not a practical Crazyflie controller. |
| 8 | [UniPi](https://openreview.net/forum?id=bo8q5MRcwy) | NeurIPS 2023 precursor that plans through text-conditioned future video and extracts actions afterward. | Historically important, but farther from a real-time recurrent aerial policy than latent auxiliary training. |

### Critical non-WAM comparator

[SINGER](https://arxiv.org/abs/2509.18610) is more directly relevant to FlightRL than the large
WAMs: it combines semantic perception with a compact navigation policy and reports asynchronous
perception around 12 Hz on Jetson Orin Nano and policy inference at 20 Hz. However, its real trials
used a five-inch cinewhoop, began with the target visible, and reported 16.67% collisions. It is
therefore evidence for compact supervised semantic control and explicit simulation, not evidence
that FlightRL's outside-FOV live gate is already solved.

## Scale and latency audit

### Verified paper facts

- [WorldFly](https://arxiv.org/pdf/2606.06147) reports four A100 GPUs for training, 14.6 GB on one
  A100, and both 7.81 seconds per step and approximately 0.5 Hz. Those two timing claims are
  arithmetically inconsistent; either interpretation is far below FlightRL's live loop.
- [NavWAM](https://arxiv.org/pdf/2606.13494) reports four RTX PRO 6000 GPUs for training and
  205.7 ms per action on one RTX PRO 6000.
- [MotionWAM](https://arxiv.org/abs/2606.09215) reports about 4.9 action chunks per second on an
  A100.
- [EgoWAM](https://gatech-rl2.github.io/egowam.github.io/EgoWAM.pdf) reports 30 Hz action-only
  inference on an RTX 4090 after removing the world head.

### FlightRL inference

The current v59 policy report records **80,997 parameters**. A 2B-parameter WAM is about 25,000
times larger; a 14B model is about 173,000 times larger. Even one byte per parameter would require
about 2 GB for a 2B model before activations. Full published WAMs are therefore infeasible on the
current AI-deck lane.

Offboard inference is technically possible only as a slow, high-level semantic planner or shadow
observer on a large GPU. It must not sit in the firmware stabilization, takeoff, landing, abort,
position/altitude-hold, or live 8 deg/s yaw path. Network latency and jitter would also make it a
poor dependency for the first semantic demo.

## Data feasibility

### Verified local facts

FlightRL can already emit aligned simulator tuples containing observation history, executed
action, next observation, and privileged state. That is enough to train a small
action-conditioned latent predictor without collecting physical-flight data. There is no
matching real v59 action/outcome dataset; a passive shadow log does not provide counterfactual or
action-conditioned outcomes.

### FlightRL inference

Raw transition count is not the primary blocker. The larger risks are simulator-to-camera domain
shift, mismatch between coarse published WAM action spaces and the 8 deg/s live cap, and learning
a predictive representation that is irrelevant to door-search decisions. Physical data
collection is neither necessary nor authorized for an initial screen.

## One later causal experiment

**Name:** WAM-inspired auxiliary next-latent screen  
**Timing:** Only after the 8 deg/s mismatch is resolved and a measured
representation/predictive-memory failure remains. The matched DAgger result is
already negative, and a shadow alone is not sufficient.

| Contract | Preregistered choice |
|---|---|
| Hypothesis | Action-conditioned next-latent prediction improves outside-FOV recovery by forcing the recurrent trunk to preserve action-relevant visual dynamics. |
| Parent | The frozen T61 training recipe and v53 perception-only parent, with the exact corrected observation, recurrence, previous-action, yaw-sign, and yaw-scale contracts. Both control and treatment must restart from identical initial controller tensors. |
| Control | Retrain the frozen pure-BC recipe with the same optimizer, stream, samples, batches, seed, and evaluation, with no auxiliary head. |
| Single controlled variable | Treatment adds one training-only head that predicts the stop-gradient next visual latent from the recurrent state and executed action. |
| Loss | One preregistered standardized latent loss with weight λ=1; no sweep and no raw-pixel target. |
| Deployment | Delete the head after training. Actor inputs, outputs, parameter count, and live inference path remain identical to the control. |
| Seeds | 11, 23, and 47 when computationally practical. |
| Evaluation | Mission success, outside-FOV success, collision, masked-camera behavior, temporal-order and action-shuffle ablations, worst lighting/noise/camera-latency groups, latency, and throughput. |
| Promotion rule | At least +5 completion points over the matched control, collision ≤3%, and no camera-causality, temporal-order, worst-group, latency, or throughput regression. |
| Stop rule | If the matched experiment fails, stop this lane; do not escalate to video diffusion or a larger WAM. |

The experiment changes one causal variable—the presence of the auxiliary next-latent objective.
Its head is training-only, so a positive result would test representation learning without
weakening the existing bounded live gate. The completed DAgger screen did not
support corrective aggregation as the current remedy; neither DAgger nor a WAM
should be expanded without a newly isolated causal failure.

## Feasibility verdict

- **Tonight / first live demo:** No. WAM work would add architecture and evaluation risk without
  addressing the verified control-contract gap.
- **AI-deck deployment:** No for current published WAMs.
- **Offboard live authority:** No for the first milestone; at most a later shadow or slow planner.
- **Training-only compact auxiliary dynamics:** Plausible and causally testable later.
- **Expected impact on the morning gate:** None. Keep the existing non-actuating real-shadow-first
  command and all stop conditions unchanged.
