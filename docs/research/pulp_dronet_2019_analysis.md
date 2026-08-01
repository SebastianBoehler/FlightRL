# PULP-DroNet architecture comparison

Primary references:

- [PULP-DroNet paper](https://arxiv.org/abs/1905.04166)
- [PULP-DroNet implementation](https://github.com/pulp-platform/pulp-dronet)

This is a literature comparison, not an implementation or promotion record.
FlightRL does not contain a current PULP-DroNet checkpoint and does not claim
reproduction of its flight results.

## What transfers directly

PULP-DroNet demonstrates the useful systems boundary for a nano-UAV: compact
onboard visual inference produces navigation intent, while the flight MCU keeps
state estimation, stabilization, and motor mixing. Its visual outputs are
translated into bounded command-level behavior rather than driving motors
directly.

FlightRL keeps that boundary:

```text
AI Deck camera + telemetry + target ID
                 |
                 v
       small recurrent actor
                 |
        bounded proposal over CPX
                 v
STM32 freshness / limits / estimator / deadman
                 |
                 v
       stock stabilizer and mixer
```

Other durable lessons are to calibrate with the real camera, quantize only
after freezing preprocessing and operators, measure target memory/latency, and
evaluate the whole closed loop rather than only classifier loss.

## Where FlightRL intentionally differs

The exact target is `aideck-navigation-policy-v3`, not a DroNet clone. It takes
one `64x48` gray4 frame, 19 normalized telemetry values, and one of three target
IDs. It proposes body-frame `vx`/`vy`, world-up `vz`, and yaw rate. The target
conditioning and local mission behavior therefore require a different training
and evaluation contract from collision-probability plus steering.

The current PyTorch edge actor is only a reference graph. FlightRL has no
current float-C/int8/GAP8 implementation, CPX proposal runtime, STM32 proposal
safety layer, or deployable checkpoint. PULP-DroNet's successful deployment is
evidence that the board class is plausible; it is not evidence that this graph
fits or runs correctly.

## Practical implications

- Keep privileged geometry and analytic controllers on the Mac as teachers;
  never expose their privileged state to the edge student.
- Train and evaluate the exact target-conditioned observation/action contract,
  rather than reviving old range-only, raw-action, or two-output actors.
- Preserve the STM32 as safety and flight-control authority.
- Treat camera capture integrity, real positive/hard-negative perception,
  recurrent sequence parity, int8 quality, ELF memory, sustained latency, and
  stale/reordered-packet rejection as independent gates.
- Use ranger/altitude data as an independent safety input when implemented; do
  not silently expand the learned actor ABI.

## Non-conclusions

The paper does not justify direct motor policies, generic checkpoint loading,
host inference as an onboard result, or deployment after a simulator teacher
passes. It also does not choose FlightRL's resolution, target vocabulary,
mission metric, quantization calibration set, or command limits.

The correct next use of this literature is as an engineering checklist while
building the exact edge-v3 lowering and STM32 boundary, not as a reason to keep
a parallel legacy avoidance stack.
