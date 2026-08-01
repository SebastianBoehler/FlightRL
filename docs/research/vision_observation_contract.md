# Vision observation decision and current status

The 2026-07-24 exploration established that a current grayscale appearance
frame must remain available; frame-difference-only input loses static scene and
target information. The deployment ABI has since been superseded and is now
defined exclusively by `aideck-navigation-policy-v3` in
`docs/edge_navigation_v3.md`.

The retired multi-channel temporal visual actor is not a valid checkpoint or
input contract. Do not recreate it from this note.

## Exact edge-v3 visual input

The actor receives one current `64x48` gray4 image:

- row-major pixels;
- first pixel in the high nibble and second in the low nibble;
- unpacked values normalized as `float32(nibble) / 15`;
- exactly 3,072 visual values before telemetry and target conditioning.

This is the model contract, not proof that the camera capture, resize, nibble
packing, transport, or deployed C implementation is correct. Those stages must
be frozen, hashed, and checked with shared byte-level test vectors.

Temporal context belongs to the actor's recurrent state. Extra appearance,
difference, motion-mask, or optical-flow channels are research variants only;
adopting one requires a new versioned contract and a complete budget/parity
rerun.

## Why one current frame

A current frame retains static door, monitor, sink, wall, and obstacle cues when
the drone stops. A signed difference or thresholded motion mask may help expose
motion, but it also responds to exposure changes and camera motion and can make
a stationary target disappear. A frame-camera difference is not an event-camera
measurement.

The small current-frame contract minimizes input transport and activation cost
while leaving short temporal inference to the recurrent actor. Its quality must
still be established through controlled held-out evaluation.

## Required capture evidence

Training data is eligible only when its source is registered and its integrity
is established. A successful decoder and a zero receiver drop counter do not
prove that frames are untorn, correctly ordered, or paired with the right
telemetry.

For each capture, retain at least:

- firmware image and source identities plus hashes;
- transport, dimensions, pixel encoding, and decode contract;
- frame indices and host timestamps;
- dropped/incomplete/error status;
- visual integrity review and evidence;
- telemetry source and synchronization method;
- physical scene/target labels and dataset split identity.

The current UDP receiver assembles payload chunks but has no end-to-end source,
frame-sequence, or checksum proof. Treat new UDP captures as `unreviewed` until
an independent integrity procedure marks them `frame_safe`. Do not admit
`unreviewed` or `known_corrupt` data to a promotion dataset. `frame_safe` is a
necessary transport property, not proof of current labels, split freshness,
telemetry synchronization, policy compatibility, or promotion eligibility.

## Experiments that remain valid

With data, seeds, policy budget, action contract, and evaluation scenes held
fixed, compare:

- source resolution and resize method;
- 8-bit archival input versus gray4 deployment preprocessing;
- exposure, blur, noise, lighting, and target scale;
- current-frame baseline versus explicitly versioned temporal alternatives;
- target-present, target-absent, and physically verified hard-negative scenes;
- stale, repeated, missing, corrupt, and reordered frame behavior.

Select from closed-loop mission success, collision/clearance, perception recall
and false positives, action smoothness, stale-input behavior, and measured host
and GAP8 cost. Offline loss or transport frame rate alone cannot select the
representation.

## Literature context

- [PULP-Dronet v3 dataset](https://zenodo.org/records/13348430) uses real
  Crazyflie/AI Deck grayscale capture and supports calibrating on the target
  camera.
- [Tiny-PULP-Dronet v3](https://arxiv.org/abs/2407.12675) shows that compact
  grayscale models can run on GAP8-class hardware, but it does not prove this
  actor's memory, latency, or task quality.
- [NanoFlowNet AI Deck example](https://github.com/gemenerik/gap8-obstacle-avoidance)
  is useful operator/tooling evidence for a possible optical-flow experiment,
  not part of the edge-v3 ABI.

All historical visual checkpoints and their run-specific claims are excluded
from active promotion context. Git history and the evidence archive preserve
lineage without granting authority.
