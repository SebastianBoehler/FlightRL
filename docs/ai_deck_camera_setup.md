# AI Deck camera capture

This guide covers camera/telemetry capture and offline perception work. It does
not load a navigation checkpoint, apply policy proposals, arm, or fly. The
canonical onboard policy contract is `docs/edge_navigation_v3.md`.

The AI Deck camera is owned by GAP8 and frames travel through the ESP32/Wi-Fi
path. The bounded paired lane logs Crazyflie state separately over `usb://0`.
Host timestamps help align streams but do not prove device synchronization.

## Hardware profile

For the reviewed Crazyflie 2.1 Brushless stack, use the explicit profile:

```text
configs/hardware/crazyflie_2_1_brushless_aideck_flow2.toml
```

It expects the AI Deck, Flow Deck v2, and Z-ranger and does not expect a
Multi-ranger. Never reuse this profile for a different physical deck stack.

With propellers removed, battery charged, and the aircraft stationary, validate
command/configuration parsing first:

```bash
python scripts/crazyflie_bringup.py \
  --config configs/hardware/crazyflie_2_1_brushless_aideck_flow2.toml \
  --dry-run check
```

Use only the bounded preflight below for live deck/TOC inspection; the paired
runner later gates its first live battery row. The dry run establishes neither.

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
PNGs. It records the source width, height, depth, and format. A 64x48 gray4
frame whose decoded values are all multiples of 17 is byte-compatible with the
edge-v3 visual segment after `float32(frame) / 255`; no upscaling is required.
The utility still records that no full edge-v3 observation or policy output was
constructed and that firmware identity was not established.

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

## Bounded stationary paired-capture gate

The legacy `aideck_udp_ground_gate.sh radio` path is not the current paired
capture path: it used a broad multi-block Crazyradio profile and could wait
forever if cflib hung during disconnect. Use the process-isolated USB path,
which records one five-variable log block and force-reaps a stuck child after a
bounded cleanup interval:

```bash
python scripts/capture_aideck_with_telemetry.py \
  --config configs/hardware/crazyflie_2_1_brushless_aideck_flow2_usb_capture.toml \
  --run-dir artifacts/ai_deck/paired_stationary_dry_run \
  --duration-s 23 \
  --frames 1200 \
  --dry-run
```

Before a real run, remove the propellers and rigidly support the aircraft over
a textured surface. The bounded preflight verifies the exact deck/TOC contract,
then plays Glass immediately before its six-second Flow interval. Gently
translate only the supported aircraft after Glass; Hero means pass and Basso
means reject. Use the USB profile when the cable has enough slack:

```bash
PREFLIGHT_DIR="artifacts/ai_deck/flow_preflight_$(date +%Y%m%dT%H%M%S)"
python scripts/run_aideck_flow_preflight.py \
  --config configs/hardware/crazyflie_2_1_brushless_aideck_flow2_usb_flow_preflight.toml \
  --run-dir "$PREFLIGHT_DIR"
```

If the USB cable prevents holding the supported aircraft low enough for a
reliable Flow/Z-ranger check, unplug USB and use the exact Crazyradio profile
instead. This alternative changes only the read-only preflight transport; it
does not authorize control or replace the USB transport used by the paired
capture:

```bash
PREFLIGHT_DIR="artifacts/ai_deck/flow_preflight_radio_$(date +%Y%m%dT%H%M%S)"
python scripts/run_aideck_flow_preflight.py \
  --config configs/hardware/crazyflie_2_1_brushless_aideck_flow2_radio_flow_preflight.toml \
  --run-dir "$PREFLIGHT_DIR"
```

The report passes only when at least 80% of all rows retain the official
PMW3901 motion/status byte `motion.motion == 0xB0` together with
`motion.squal >= 80`, deliberate motion produces nonzero deltas, Z-ranger
remains plausible, children exit cleanly, and no packet loss is logged. It
expires after 15 minutes.

After it passes, do not reseat or change the deck stack. If radio was used,
place the aircraft on its stationary support and reconnect USB without moving
the decks. Wait for the carried estimator state to settle before capture.
Connect the Mac to the AI Deck AP so its route is Mac `192.168.4.2` to deck
`192.168.4.1`; the paired telemetry path is always `usb://0`. Fail before
capture if either route is unavailable.

Keep the supported aircraft stationary and point the camera at a static,
high-contrast scene. This transport run is not a semantic dataset. The paired
runner checks the first telemetry row and battery before playing Glass and
starting the camera; Hero/Basso marks its bounded end:

```bash
RUN_DIR="artifacts/ai_deck/paired_stationary_$(date +%Y%m%dT%H%M%S)"
if python scripts/capture_aideck_with_telemetry.py \
    --config configs/hardware/crazyflie_2_1_brushless_aideck_flow2_usb_capture.toml \
    --run-dir "$RUN_DIR" --duration-s 23 --frames 1200 \
    --flow-preflight-report "$PREFLIGHT_DIR/preflight_process.json"; then
  python scripts/validate_aideck_paired_capture.py "$RUN_DIR"
else
  exit 2
fi
```

Stop and reject the run on any process timeout/nonzero exit, packet-loss
message, invalid Flow status, low Flow quality, rejected camera datagram,
camera drop fraction above 0.5%, camera rate outside 55--75 Hz, telemetry gap
above 75 ms, any camera host-time gap above 75 ms, nearest camera/telemetry
host-time gap above 30 ms, incomplete stream overlap, battery below 3.70 V, or
stationary span above 12 cm X, 5 cm Y, 2 cm Z, or 1.5 degrees yaw. Passing this
gate establishes bounded stationary transport and host-time proximity only.
UDP sequence/checksum integrity, device-clock synchronization, semantic
generalization, training eligibility, deployment, shadow, and flight authority
remain false.

## Non-actuating semantic checks

Offline grounding works directly on archived PNG/JPEG frames or a decoded
capture NPZ, preserving the NPZ frame indices and capture metadata:

```bash
python scripts/evaluate_aideck_grounding.py \
  artifacts/ai_deck/capture/decoded_frames.npz \
  --prompt monitor \
  --output artifacts/semantic/offline-monitor \
  --require-detection
```

For an operator-labeled positive/negative pair already captured as exact 64x48
gray4, measure structural scene separability under the edge-v3 pixel contract:

```bash
python scripts/evaluate_aideck_pair.py \
  --positive artifacts/ai_deck/door_positive/decoded_frames.npz \
  --negative artifacts/ai_deck/monitor_negative/decoded_frames.npz \
  --sample-count 120 \
  --output artifacts/semantic/door_positive_vs_monitor_negative.json
```

This two-fold calibration removes each frame's global brightness and contrast
before nearest-centroid evaluation and also runs a spatially destroyed
gray-level-histogram baseline. If the histogram also separates the pair, the
result is session/scene separability, not semantic calibration. A pass means
only that the two named operator-labeled scenes are observably different. It is
not a category-level door detector, independent holdout, or training authority.

The reviewed centered-door and monitor captures are exactly such a confounded
pair: both the structural classifier and the histogram baseline separate them.
The current grounding stack proposes boxes in nearly every sampled frame of
both clips and its verifier rejects all of them. Lowering thresholds would
therefore increase false positives rather than repair a decoder mismatch.

The camera-only streaming utility also never commands the aircraft:

```bash
python scripts/crazyflie_semantic_find.py \
  --prompt monitor \
  --duration-s 30 \
  --output artifacts/semantic/camera-only-monitor
```

These are host-side perception/capture reports. Exact gray4 frames can match the
edge-v3 visual segment, but the captures do not contain the complete 19-value
telemetry chronology, target token, reset/action history, or a retained accepted
checkpoint. They do not prove target-conditioned navigation or grant
shadow/flight authority. Hard negatives must be physically verified.

## Recognizable coverage behavior

The semantic-free alternative is a simulation-only scan--advance patrol. Its
actor contract consumes exact 64x48 gray4 plus the same 19 telemetry values,
has no target token, object detector, range, map, or privileged pose, and can
command only forward speed and yaw within 0.25 m/s and 8 degrees/s. The current
privileged teacher visibly advances, scans in a fixed direction for at least 90
degrees when its camera-aligned front clearance is blocked, and advances again
only after clearance recovers:

```bash
python scripts/evaluate_scan_advance_behavior.py \
  --seed-start 512 --episodes 4 --steps 1800 \
  --output artifacts/evidence/scan_advance_front_clearance_36s.json

python scripts/render_scan_advance_demo.py \
  --seed 515 --steps 1800 \
  --output artifacts/evidence/scan_advance_front_clearance_demo_seed515.mp4
```

This teacher uses a privileged front-range value only for simulation labels; it
is not deployable on the reviewed aircraft, which has no Multi-ranger. A small
camera student passes a real-rendered matched-pair causal smoke, but fails the
held-out closed-loop camera test: frozen images produce identical path and
coverage, while permuted histories do better. At the full 1,800-step horizon,
the clean actor collides in both held-out rooms after reaching a front-aligned
obstacle challenge. It moves recognizably, but is neither camera-causal nor
avoidance-ready. Simulation success grants no hardware authority.

## Edge-v3 camera requirements

The exact gray4 resize, high-nibble quantization, packing, decode, and visual
tensor conversion now have shared host/native regression coverage. Before a
camera policy can support the onboard actor, still add and verify:

1. authoritative flashed-firmware identity plus frame sequence/checksum proof;
2. full 3,091-value coverage observation or 3,094-value navigation observation
   chronology, including exact reset and previous-action semantics;
3. behavior for missing, stale, duplicate, reordered, and corrupt frames;
4. recurrent reset propagation after invalid input or excessive frame gap;
5. sustained latency and memory under camera, inference, CPX, and radio load;
6. fresh positive, target-absent, hard-negative, lighting, blur, and motion
   evaluation under the exact frozen firmware and model contract.

Until then, AI Deck work remains capture, offline replay, or non-actuating
inference measurement.
