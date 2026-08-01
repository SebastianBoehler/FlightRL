# Edge navigation v3

## Decision and boundary

The next policy should be trained on the Mac as an **edge-shaped executable
PyTorch reference**, then frozen, lowered, quantized, and checked against its
GAP8 implementation. The current PyTorch module is not yet the exact deployment
graph: float-C kernels, an int8 reference, GAP8 kernels, a CPX setpoint ABI, and
the STM32 safety application do not exist.

All pre-v3 navigation checkpoints were removed from the active workspace. They
are not valid parents for this graph, and no existing checkpoint has v3 flight
authority.

The machine-readable contract is `aideck-navigation-policy-v3` in
`src/flightrl/puffer4_edge_contract.py`.

```text
Mac: simulator + privileged teacher + critic + PyTorch actor reference
                              |
               freeze operators and preprocessing
                              v
future tooling: float C -> calibrated int8 C -> sequence parity
                              |
                              v
AI Deck GAP8: camera + target-conditioned CNN/recurrent inference
                              |
                  normalized proposal over CPX
                              v
STM32: freshness + clamps + slew + geofence + estimator checks + deadman
                              |
                 stock estimator/stabilizer -> motors
```

## Target conditioning

The closed vocabulary is deliberately limited to the simulator's implemented
categories:

| `target_id` | canonical target |
| ---: | --- |
| 0 | `door` |
| 1 | `monitor` |
| 2 | `sink` |

Free-form text is never an onboard policy input. A validated mission compiler
must map a target to one of these exact IDs before the mission starts; all other
IDs are rejected. Vocabulary membership does not mean a class has passed real
perception or navigation gates.

The retained desktop semantic mission compiler emits scene-local object
indices and supports categories such as `desk`; it is not this edge-v3 mission
compiler and must not feed the wire contract. Edge integration requires an
explicit semantic object-ID-to-approved-target-ID binding with rejection of
unsupported categories.

The one shared CNN receives the current frame once. The active target one-hot
multiplicatively gates its visual features before the grounding head, so target
selection changes which image features drive visibility and box output rather
than merely adding a class bias. The actor also receives the target one-hot.
Grounding labels describe only the active target: `visible=0` when that target
is absent, and center/scale losses are masked in that case. Another visible
class must not create a positive auxiliary label for the active target.

## Model observation

The model tensor is `float32[3094]` in this exact order:

1. 3,072 row-major gray4 pixels. The first pixel is the high nibble and the
   second is the low nibble; each unpacked nibble becomes `float32(nibble)/15`.
2. 19 normalized telemetry values in the order below.
3. Three float32 one-hot target values indexed by `target_id`.

All clips are inclusive. Body axes are FLU: `+x` forward, `+y` left, `+z` up.

| values | physical source | normalization |
| --- | --- | --- |
| body velocity x/y/z | m/s, current body FLU | divide by `1.0/1.0/0.5`, clip `[-1,1]` |
| body rate x/y/z | rad/s, right-hand body FLU | divide by `6/6/4`, clip `[-1,1]` |
| body-up x/y/z | world-up unit vector in body FLU | scale 1, clip `[-1,1]` |
| altitude | m world-up from takeoff origin | divide by `2.5`, clip `[0,1]` |
| origin displacement fwd/left/up | m in mission-start yaw frame/world-up | divide by `4/4/2`, clip `[-1,1]` |
| relative yaw sin/cos | current minus mission-start yaw | scale 1, clip `[-1,1]` |
| previous vx/vy/vz/yaw-rate | last setpoint actually applied by STM32 | divide by `0.25/0.25/0.15 m/s` and `45 deg/s`, clip `[-1,1]` |

The previous action is not the raw actor proposal. It is post-safety feedback
from the STM32, expressed in the frames of the step where it was applied. It is
all zeros before the first applied setpoint after a reset.

## Canonical wire records

The contract defines little-endian, packed, no-padding reference records so
replay, host C, GAP8, and STM32 tests decode the same bytes. The 1,635-byte input
record is a canonical serialization; it is not a claim that the local camera
frame will be sent as one CPX packet.

| input offset | type | field |
| ---: | --- | --- |
| 0 | `uint8` | protocol version `3` |
| 1 | `uint8` | flags; bit 0 means reset state, all others zero |
| 2 | `uint32` | wrap-aware frame sequence |
| 6 | `uint32` | wrap-aware monotonic capture time in microseconds |
| 10 | `uint32` | wrap-aware monotonic telemetry time in microseconds |
| 14 | `uint32` | mission epoch |
| 18 | `uint32` | arming epoch |
| 22 | `uint8` | target ID |
| 23 | `float32[19]` | normalized telemetry in contract order |
| 99 | `uint8[1536]` | packed current gray4 frame |

The 39-byte proposal record contains version/flags, source frame sequence,
source capture time, mission and arming epochs, source target ID, proposal
sequence, then normalized `float32[4]` action.
Invalid, incomplete, duplicate, or reordered records are rejected.

## Action and reset semantics

Each normalized output is clipped to `[-1,1]` before this mapping:

| output | physical proposal | positive direction |
| --- | ---: | --- |
| `vx` | `u * 0.25 m/s` | body forward |
| `vy` | `u * 0.25 m/s` | body left |
| `vz` | `u * 0.15 m/s` | world up |
| `yaw_rate` | `u * 45 deg/s` | left/CCW about world up |

These are proposal scales, not permission to bypass tighter STM32 limits.

The hidden state is zeroed before inference when reset bit 0 is set. Reset is
mandatory on actor boot, mission start or target change, arming epoch change,
estimator/origin reset, after an invalid record, or when successive accepted
capture times are more than 100 ms apart. A rejected record never updates state;
the next valid record must request reset. State commits only after successful
inference.

## Quantitative budget

For hidden size 48, the current reference has:

- 17,700 parameters;
- 17,336 prospective int8 weight bytes plus 1,456 int32 bias bytes;
- 18,792 total prospective quantized parameter bytes;
- 96,144 estimated multiply-accumulates per step;
- 3,094 model-input elements and 1,536 elements in the largest internal
  activation (3,094 elements for the largest single tensor);
- a 1,635-byte canonical input record.

These are static PyTorch graph estimates. They exclude lowering choices,
quantization metadata, kernels, stack, im2col/work buffers, CPX buffers, ELF
sections, and measured latency. Only an actual GAP8 ELF map and target run can
prove deployment fit.

## Required parity and promotion gates

Before this reference may be called the exact deployment graph:

1. Freeze and hash preprocessing, operator lowering, weights, calibration data,
   activation scales, rounding, saturation, and tensor layouts.
2. Match PyTorch float and float C over complete recurrent sequences, including
   every reset/error case, to maximum absolute error `1e-5` for action,
   grounding, and hidden state.
3. Pass int8 calibration and held-out mission regression; quantization quality
   is an evaluation gate, not mislabeled float/int8 equality.
4. Match host int8 C and GAP8 bit-for-bit for action, grounding, and hidden state
   after every valid step.
5. Prove ELF L1/L2/stack/workspace and sustained latency on the GAP8.
6. Define and verify a trusted STM32-side `goal_reached` input. The current
   proposal record carries no target pose, grounding result, or arrival claim,
   so it cannot yet advance a mission from navigate to hold or land.
7. Prove CPX sequencing/freshness and independent STM32 clamps/deadman, then
   progress through capture, replay, shadow, and tethered bounded-axis gates.

Until those gates exist and pass, hardware work remains camera capture,
telemetry, and policy shadow only.
