# Fixed-door observability gate

## Decision

Do not start fixed-door reinforcement learning until a compact camera head can
detect and localize a door without receiving target coordinates. The policy
input is the native AI Deck contract: one `64x48` 4-bit grayscale frame.
MuJoCo segmentation supplies training labels only.

The gate requires:

- held-out synthetic visibility AUROC at least `0.90`;
- held-out median centroid error at most `0.12` image widths;
- real door-positive recall at least `0.80`;
- real door-negative false-positive rate at most `0.10`;
- real median centroid error at most `0.12` image widths.

## Frame integrity

The camera, deck, and prop guards are rigid. Multiple guard positions or two
room views inside one decoded image indicate a raster-wrapped transport frame,
not physical motion. The pre-fix capture at
`artifacts/semantic/live-flight-monitor-qvga-clip-20260726/frames` is therefore
registered as `known_corrupt`.

Post-fix three-buffer captures keep complete sensor-frame boundaries and drop
whole frames under backpressure. Reviewed datasets are listed in
`configs/semantic/aideck_frame_integrity.json`. Real-evidence loading rejects
corrupt, unreviewed, and unregistered captures.

## Results

The original simulator rendered a door as a plain dark rectangle, visually
confusable with cabinets, wall edges, partitions, and monitors.

| Condition | Train rooms | Held-out rooms | AUROC | Centroid error | Result |
|---|---:|---:|---:|---:|---|
| Flat door, 9,136 parameters | 24 | 8 | `0.7202` | `0.1012` | fail |
| Structured door, 9,136 parameters | 24 | 8 | `0.7993` | `0.1102` | fail |
| Structured door, 34,772 parameters | 24 | 8 | `0.7924` | `0.1076` | fail |
| Structured door, 34,772 parameters | 128 | 32 | `0.9127` | `0.0912` | synthetic pass |

The 24-room model reached `0.9644` AUROC on its training rooms but only
`0.7924` on disjoint rooms. This identified room-seed overfitting rather than
insufficient model capacity. Increasing room diversity, while holding the
model and camera contract fixed, passed the synthetic gate.

The accepted synthetic checkpoint is:

```text
artifacts/semantic/door-observability-64x48-r128-20260729/door_observability.pt
SHA-256 66474796b978fe484362d554a59dd071496dbc6abb6ed50ab66f13cf6f89cffc
```

Two post-fix, handheld QVGA sweeps supplied real evidence:

- `door-observability-positive-20260729-run1`: 300 frames spanning complete,
  partial, near, far, centered, and edge views of the room door;
- `door-observability-negative-20260729-run1`: 300 door-absent frames with
  oven, cabinets, counters, and other rectangular distractors.

Both captures have fixed prop-guard geometry and coherent room content. The
positive capture had one whole-frame sequence drop; the negative capture had
none. Neither has raster wrapping or mixed sensor frames.

Fifteen manually reviewed positive boxes and thirty negative frames form the
real manifest. The first temporal third of each class calibrates the visibility
threshold; the remaining thirty samples are held out from calibration.

| Real held-out condition | Samples | Recall | False-positive rate | Centroid error | Result |
|---|---:|---:|---:|---:|---|
| Door-positive | 10 | `0.90` | n/a | `0.1113` widths | pass |
| Door-negative | 20 | n/a | `0.00` | n/a | pass |

The calibrated threshold is `0.1427`. Across all reviewed data, positive scores
range from `0.1325` to `0.9345`; negative scores range from `0.0` to `0.0199`.
This is evidence of real-domain confidence shift, but the classes remain
strongly separated.

The combined synthetic and real observability gate reports `passed`:

```text
artifacts/semantic/door-observability-real-gate-20260729/report.json
```

## Native renderer reconciliation

The original result above remains valid for the MuJoCo-trained checkpoint and
the evidence available when it was accepted. It does not transfer unchanged to
the later native renderer. After adding rendered door masks, multi-seed replay,
independent appearance seeds, and rectangular distractors, no single compact
head passed both the current native and real gates:

| Head | Native AUROC | Native centroid | Real recall | Real FPR | Real centroid | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Original real-validated head | `0.6546` | `0.1227` | `0.90` | `0.00` | `0.1113` | Real pass, current native fail |
| Native v24 | `0.9741` | `0.0643` | `0.90` | `0.25` | `0.1450` | Real fail |
| Native v26 plus real calibration replay | `0.9739` | `0.0650` | `1.00` | `0.20` | `0.0648` | Real fail |
| Native v27 plus richer distractors | `0.9371` | `0.0857` | `1.00` | `0.15` | `0.0997` | Real fail |
| Preselected anchored screen | `0.9395` | `0.0993` | `1.00` | `0.20` | `0.1472` | Real fail; regularizer removed |

A calibration-only hypothesis was also rejected. Raising the threshold to the
highest calibration-only value preserving `0.80` recall removed false
positives but reduced held-out recall to `0.40`. The production calibration
method was restored rather than weakening the gate.

The real manifest has now been inspected after several model changes. It is
useful development evidence, but it is no longer an untouched final holdout.
Any later promotion requires a newly captured and pre-registered positive and
negative sequence that is evaluated once after model selection.

This is the intended result of the staged research process: it identified a
renderer-to-camera appearance gap before control training could turn a
perception shortcut into apparent mission progress.

## Next gate

Do not train the fixed-door control policy yet. First create a cross-domain
grounder candidate without using the current development manifest for stopping:

1. Expand native scene composition beyond one wall rectangle and one box so
   cabinets, counters, windows, monitors, openings, occlusion, and multiple
   simultaneous distractors occur in training.
2. Keep the original real-validated head as a frozen comparison, not an
   initialization that is assumed to remain valid after native fine-tuning.
3. Select the new head only on disjoint native grammars and a separate real
   calibration split.
4. Evaluate once on a fresh AI Deck capture. Only a combined pass unlocks
   recurrent DAgger and PPO control training.

Live operation remains shadow-only after the later simulation policy gate
passes; observability alone never grants flight authority.
