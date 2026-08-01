# FlightRL experiment lineage and control protocol

Date: 2026-07-30

This document audits whether the experiment sequence made sense and defines the
flow for future policy versions.

## Chronological lineage audit

| Phase | Question | Verdict | Reason |
| --- | --- | --- | --- |
| Deckless baseline | Can training and flight continue without the broken Ranger Deck? | Necessary and successful | Produced modular observations and preserved firmware stabilization |
| Direct raw transfer | Can the simulation policy directly own high-authority control? | Premature but informative | Exposed pitch-sign and drift-distribution defects; justified bounded authority |
| AI Deck recovery | Is camera hardware available and stable? | Necessary and successful | JTAG recovery removed a real hardware blocker |
| Camera FPS work | Can policy frames and telemetry be transported fast enough? | Necessary through stable 65 FPS | Complete-frame transport and timing were prerequisites; 100+ FPS was a useful ceiling test |
| Native local visual policy | Can a small recurrent camera policy learn local control? | Strong, coherent experiment | High throughput, domain randomization, causal masking, and held-out scenes answered one question |
| Repeated waypoint flights | Does the real bounded control contract work? | Useful, then diminishing return | First runs isolated hardware and safety; later deterministic-forward repeats did not test semantics |
| Broad MuJoCo semantics | Can grounding, mapping, exploration, safety, and approach work together? | Over-integrated and weakly controlled | Too many simultaneous changes and about 270 SPS made failures hard to attribute |
| Door grounder scaling | Is perception limited by model size or scene diversity? | Strong, coherent experiment | Fixed capacity plus 24-to-128-room scaling isolated overfitting |
| Native/real grounder gate | Does synthetic perception transfer? | Strong negative result | Caught renderer gap and prevented proxy metrics from authorizing flight |
| Fixed-door v45-v58 | Can the recurrent student learn the teacher behavior? | Partly informative, too much version churn | Phase/evidence and PPO effects were learned, but the old controller initialization remained confounded |
| Fresh-control v59 | Was controller initialization the main blocker? | Decisive | One structural intervention changed success from single digits to 79.11% with low collision |
| Next live gate | Does v59 transfer enough to control yaw from real frames? | Not run | This is now the critical path |

## Does the lineage make sense?

The high-level direction makes sense:

1. stabilize hardware and observation contracts;
2. prove low-resolution visual control;
3. validate bounded real authority;
4. isolate one semantic category;
5. train a privileged teacher and deployable recurrent student;
6. stage shadow, one-axis flight, and approach authority.

The execution became inefficient in two regions:

- The broad semantic lane combined several unsolved components before any one
  had a stable contract.
- Versions v45-v58 kept training an inherited controller while changing
  imitation, PPO, detector, and observation details. The decisive
  fresh-controller test should have occurred earlier.

The project was not going nowhere, but it sometimes optimized the nearest
failing metric instead of identifying which assumption generated the failure.
The v59 reset is evidence that periodically returning to architecture and
observability was more valuable than another local hyperparameter change.

## Required experiment card

Before starting a training version, record:

- hypothesis in one sentence;
- frozen baseline artifact and SHA;
- exactly one primary intervention;
- invariants that must not change;
- train and evaluation seeds;
- environment-transition and optimizer-update budgets;
- scene-distribution ID and held-out-distribution ID;
- observation and action contract hashes;
- primary metric and risk metrics;
- promotion threshold and kill condition;
- expected runtime.

If more than one primary intervention is required, split the work into a
factorial comparison or sequential experiments.

## Standard flow

1. **Contract freeze:** verify camera tensor, nonvisual observations, action
   scale/sign, recurrence reset, phase logic, and previous executed action.
2. **Mechanism test:** run unit tests and a small deterministic scenario that
   directly exercises the intervention. Do not promote from this screen.
3. **Equal-budget screen:** compare baseline and intervention on the same three
   seeds and episode-indexed procedural streams.
4. **Full held-out evaluation:** use at least 1,000 episodes per seed and report
   aggregate plus worst factor groups.
5. **Causal ablations:** mask current image and camera-derived evidence, shuffle
   temporal order, and compare an oracle upper bound where relevant.
6. **Independent challenge:** replay in MuJoCo or a richer visual lane without
   using that result for training selection.
7. **Real replay/shadow:** bind the exact checkpoint, evaluation, raw trace, and
   summary by SHA.
8. **Bounded live gate:** add one authority axis or envelope at a time.
9. **Post-run decision:** write `promote`, `retain baseline`, or `kill`, with the
   primary reason. Do not create another version until this is recorded.

## Metric hierarchy

Use metrics in this order:

1. contract correctness and finite outputs;
2. collision, abort, stale-input, and minimum-clearance risk;
3. complete mission success;
4. outside-FOV and worst-group success;
5. causal camera dependence;
6. action smoothness and time to completion;
7. environment SPS, learner SPS, parameters, MACs, and latency;
8. PPO return and auxiliary losses.

Training return cannot override a failed mission or safety metric.

## Room and duration controls

Room count and training duration are experiments, not defaults:

- compare topology seeds at 8, 32, 128, and 512 only after the architecture is
  stable;
- continuously randomize materials, light, target geometry, occlusion,
  distractors, camera parameters, latency, and dynamics;
- increase samples with distribution complexity so each factor remains covered;
- evaluate on disjoint topology grammars and factor ranges;
- run equal transition and update budgets before any longer continuation;
- promote a longer run only if the full held-out curve improves rather than the
  training return alone.

## Immediate registered experiments

### Live L0

Hypothesis: v59 produces correctly signed, fresh, finite door-search yaw actions
on real AI Deck frames.

Intervention: replace v53 shadow with exact v59 shadow. No flight.

Decision: proceed to L1 only if the hash-bound readiness artifact passes.

### Live L1

Hypothesis: v59 can search for and center a door with yaw-only authority while
firmware holds position and height.

Intervention: enable at most 8 deg/s yaw; translation remains zero.

Decision: pass only on reviewed search, acquisition, centering, stop, and log.

### Train T60

Hypothesis: on-policy DAgger improves the fresh v59 controller by covering its
own failure states.

Baseline: fresh-controller BC retrained under the corrected yaw contract.

Intervention: fresh-controller DAgger at the same budget and scene streams.

Seeds: 11, 23, 47. Promote only for at least five completion points of gain,
collision at most 3%, and no worst-group or camera-causality regression.

Obstacle randomization, additional room scaling, and PPO are separate later
experiments.
