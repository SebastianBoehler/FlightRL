# AI Deck camera capture

This guide covers camera/telemetry capture and offline perception work. It does
not load a navigation checkpoint, apply policy proposals, arm, or fly. The
canonical onboard policy contract is `docs/edge_navigation_v3.md`.

The AI Deck camera is owned by GAP8 and frames travel through the ESP32/Wi-Fi
path. Crazyflie state is logged separately over radio. Host timestamps can help
align the streams, but they are not proof of frame/telemetry synchronization.

## Hardware profile

For the reviewed Crazyflie 2.1 Brushless stack, use the explicit profile:

```text
configs/hardware/crazyflie_2_1_brushless_aideck_flow2.toml
```

It expects the AI Deck, Flow Deck v2, and Z-ranger and does not expect a
Multi-ranger. Never reuse this profile for a different physical deck stack.

With propellers removed, battery charged, and the aircraft stationary, first
validate command/configuration parsing and then inspect the connected decks:

```bash
python scripts/crazyflie_bringup.py \
  --config configs/hardware/crazyflie_2_1_brushless_aideck_flow2.toml \
  --dry-run check

python scripts/crazyflie_bringup.py \
  --config configs/hardware/crazyflie_2_1_brushless_aideck_flow2.toml \
  check
```

Stop on missing/unexpected decks, low battery, estimator/supervisor failures,
radio instability, or an unresponsive camera. A historical successful setup is
not evidence for the current physical state.

## Firmware boundary

Follow Bitcraze's [AI Deck getting-started
guide](https://www.bitcraze.io/documentation/tutorials/getting-started-with-aideck/)
for the canonical bootloader, ESP32, and GAP8 setup. FlightRL's pinned UDP
streamer tooling can inspect, prepare, and build its sources without flashing:

```bash
scripts/aideck_udp_streamer.sh status
scripts/aideck_udp_streamer.sh preflight
scripts/aideck_udp_streamer.sh build
```

Flashing is a separate destructive hardware operation guarded by explicit
confirmation tokens. Do not infer permission to flash from a capture or
software task, and do not flash an image merely because its historical
filename looks familiar. Verify the actual image hash, source/build identity,
aircraft variant, power, link, and recovery path first.

If normal GAP8 flashing is incomplete, stop. Do not loop retries. The dedicated
recovery procedure and physical connector warnings are in
`docs/ai_deck_jtag_recovery.md`.

## Decode-only frame capture

The supported capture utility decodes frames and writes an NPZ plus optional
PNGs. It explicitly records that policy outputs are absent and edge-v3
preprocessing was not applied.

Official TCP stream:

```bash
python scripts/capture_aideck_vision.py \
  --transport tcp \
  --host 192.168.4.1 \
  --frames 100 \
  --frame-dir artifacts/ai_deck/capture/frames \
  --output artifacts/ai_deck/capture/decoded_frames.npz
```

Pinned UDP streamer:

```bash
python scripts/capture_aideck_vision.py \
  --transport udp \
  --host 192.168.4.1 \
  --bind-port 5001 \
  --frames 100 \
  --frame-dir artifacts/ai_deck/capture/frames \
  --output artifacts/ai_deck/capture/decoded_frames.npz
```

Inspect `complete`, `capture_error`, `dropped_frames`, decoded shape/dtype, and
the saved frames. Partial capture is written for diagnosis and then exits with
an error; it must not be silently treated as a complete dataset.

The current UDP receiver assembles chunks by arrival but has no end-to-end
sender identity, frame sequence, or checksum. Zero decoder errors or receiver
drops therefore does not prove frame integrity. Register new captures as
`unreviewed` until a separate visual/integrity procedure marks them
`frame_safe`; never train a promotion candidate from `unreviewed` or
`known_corrupt` frames. `frame_safe` is only a transport-integrity prerequisite;
it does not certify labels, split freshness, telemetry synchronization, policy
compatibility, or promotion eligibility.

## Concurrent telemetry

Record radio telemetry to a separate file using the same explicit hardware
profile:

```bash
python scripts/crazyflie_log.py \
  --config configs/hardware/crazyflie_2_1_brushless_aideck_flow2.toml \
  --duration-s 30 \
  --output artifacts/crazyflie_logs/aideck_capture.csv \
  --console-output artifacts/crazyflie_logs/aideck_capture_console.jsonl
```

Capture frame index/time, telemetry host time, firmware/deck identities,
transport, scene/target labels, battery/supervisor state, and the exact physical
setup. Establish synchronization error before using paired samples for
supervision or latency claims.

## Non-actuating semantic checks

Offline grounding works on archived PNG/JPEG frames:

```bash
python scripts/evaluate_aideck_grounding.py \
  artifacts/ai_deck/capture/frames \
  --prompt monitor \
  --output artifacts/semantic/offline-monitor \
  --require-detection
```

The camera-only streaming utility also never commands the aircraft:

```bash
python scripts/crazyflie_semantic_find.py \
  --prompt monitor \
  --duration-s 30 \
  --output artifacts/semantic/camera-only-monitor
```

These are host-side perception/capture reports. They do not implement edge-v3
preprocessing, prove target-conditioned navigation, or grant shadow/flight
authority. Hard negatives must be physically verified; a failed/positive
detection count is meaningless if the target status of the scene is unknown.

## Edge-v3 camera requirements

Before camera data or firmware can support the onboard actor, add and verify:

1. frozen resize, grayscale, gray4 quantization, and nibble packing;
2. shared host/GAP8 byte test vectors for all 3,072 pixels;
3. frame sequence, capture time, source, payload length, and integrity checks;
4. behavior for missing, stale, duplicate, reordered, and corrupt frames;
5. recurrent reset propagation after invalid input or excessive frame gap;
6. sustained latency and memory under camera, inference, CPX, and radio load;
7. fresh positive, target-absent, hard-negative, lighting, blur, and motion
   evaluation under the exact frozen firmware and model contract.

Until then, AI Deck work remains capture, offline replay, or non-actuating
inference measurement.
