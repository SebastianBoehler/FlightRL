# AI Deck Camera Setup Notes

Source checked: 2026-07-25.

These notes are for using the Bitcraze AI deck 1.1 as a camera/perception
source for FlightRL. The deck is not a normal Crazyflie log source: the camera
is owned by the GAP8 side and images are streamed through the ESP32/WiFi path.
Crazyflie radio telemetry can log flight state at the same time, but image
frames need a separate capture and timestamp-sync path.

## Current Local State

- Stable baseline flights now work with the AI Deck above and Flow Deck v2
  below. The latest directional run held about 0.31 m altitude over a roughly
  8x9 cm footprint while the Puffer policy remained shadow-only.
- The official TCP/RAW stream is not flight-ready. One run collected 65 frames
  before takeoff and then timed out; the next collected one in-flight frame and
  timed out. A ground-only capture also froze before motor start, so payload and
  battery sag can aggravate the failure but do not fully explain it.
- The in-flight frame is preserved at
  `artifacts/ai_deck/line_frames_20260725_run6/frame-000001.png`.
- FlightRL now supports both the official TCP stream and the maintainer-linked
  LARICS UDP/JPEG workaround through `--transport tcp|udp`.
- On 2026-07-24, the GAP8 bootloader was restored over JTAG with the expected
  TAP ID `0x149511c3`. The official `2025.02` WiFi streamer then reached 100%
  over `deck-bcAI:gap8-fw`, the `WiFi streaming example` access point became
  reachable at `192.168.4.1:5000`, and a valid 324x244 Bayer frame was captured
  to
  `artifacts/ai_deck/camera_frames/aideck-frame-20260724T170046Z.png`.
  A post-capture radio check still reported `deck.bcAI=1`.
- Crazyflie radio URI seen earlier: `radio://0/80/2M`.
- Flow-only telemetry config works when the radio link is healthy.
- `cfclient` and `cfloader` exist at `/Users/sebastianboehler/.local/bin`.
- The active conda Python does not have `cfloader`, `cfclient`, or `cv2`.
- The `cfclient` uv tool Python has `cflib` and `cfclient`, but not `cv2`.
- Docker is available locally.
- No local `aideck-gap8-examples` checkout or `opencv-viewer.py` was found.
- The common AP address `192.168.4.1` did not respond during the last check.
- A later deck-param probe failed with `Too many packets lost`, so do not treat
  current deck visibility as verified until the battery is charged and the radio
  link is stable.
- On 2026-07-06, after a power-cycle, `deck.bcAI=1` and `deck.bcFlow2=1` were
  confirmed over radio. Flashing the official `2025.02` WiFi streamer with
  `cfloader flash ... deck-bcAI:gap8-fw -w radio://0/80/2M/E7E7E7E7E7` stalled
  after `Reset to bootloader mode ...` and two cache warnings, with no progress
  for several minutes. The process was interrupted. Normal scan recovered, but
  deck detection then became inconsistent (`deck.bcAI=0`, `deck.bcFlow2=0`,
  `deck.bcZRanger2=1`). Stop at this point and power-cycle before any retry.
- After a power-cycle, `deck.bcAI=1`, `deck.bcFlow2=1`, and idle battery
  `pm.vbat=4.17V` were confirmed. A second `cfloader flash` retry returned
  quickly but only printed nRF51 softdevice checks; it did not print the expected
  `Deck bcAI:gap8` or write-progress lines, and the AI-deck AP/socket at
  `192.168.4.1:5000` remained unreachable. After this retry, deck detection
  degraded to `deck.bcAI=0` while Flow/Z-ranger still reported present. Stop and
  power-cycle; do not fly or retry the over-radio GAP8 flash until AI-deck
  firmware/ESP/GAP8 bootloader state is recovered, preferably through cfclient's
  bootloader UI or the JTAG bootloader workflow from the Bitcraze tutorial.
- After reseating a moved AI-deck contact, `deck.bcAI=1`, `deck.bcFlow2=1`, and
  `deck.bcZRanger2=1` were confirmed again over radio. Passive telemetry wrote
  `artifacts/crazyflie_logs/post_ai_deck_contact_fix_check_20260706.csv` with
  582 samples, `pm.vbat=4.16V`, `batteryLevel=90`, and no flying/tumbled flags.
  The AI-deck AP/socket still did not respond at `192.168.4.1:5000`, so the
  hardware contact is recovered but the WiFi streamer app is still not running.
- After a later clean pin check and power-cycle, `deck.bcAI=1`, `deck.bcFlow2=1`,
  and `deck.bcZRanger2=1` were again confirmed. A guarded retry of the same
  over-radio GAP8 flash was left running for more than three minutes. It still
  only printed `Reset to bootloader mode ...` and two cache warnings, never the
  expected `Deck bcAI:gap8` or write-progress output. The attempt was stopped.
  Normal scan recovered, but deck detection again degraded to `deck.bcAI=0`,
  `deck.bcFlow2=0`, `deck.bcZRanger2=1`. Treat over-radio GAP8 flashing from
  this CLI setup as blocked until cfclient bootloader update or JTAG recovery is
  performed.
- The cfclient-style release zip path was tested with
  `firmware-cf2-2026.04.zip`. Targeting `deck-bcAI:esp-fw` succeeded: the log
  reached `Deck bcAI:esp, reset to bootloader` and wrote `bcAI:esp` from 0% to
  100%, then skipped GAP8 because it was not in the target list. After this,
  `deck.bcAI=1`, `deck.bcFlow2=1`, and `deck.bcZRanger2=1` were detected again.
- Retrying the GAP8 WiFi streamer app after the ESP update still stalled before
  `Deck bcAI:gap8` progress. A full `firmware-cf2-2026.04.zip` flash then wrote
  STM32 and nRF firmware but hung before deck update/reset completion. Normal
  scan recovered afterward, but firmware/deck connect checks hung until
  interrupted. Next recovery step is a physical power cycle, then inspect only.
  Do not run another radio GAP8 app flash unless GAP8 bootloader/JTAG recovery
  has been completed or cfclient UI shows a healthier path.

## Official Setup Path

Bitcraze's getting-started tutorial is the canonical setup path:
https://www.bitcraze.io/documentation/tutorials/getting-started-with-aideck/

The important steps are:

1. Use `cfclient` bootloader update with only the AI deck attached.
   This updates Crazyflie STM32, nRF51, and AI-deck ESP32 firmware.
2. Confirm in the cfclient console that the ESP32 initializes.
3. Flash the GAP8 bootloader via JTAG only if GAP8 flashing hangs around
   4% or 99%. Bitcraze states this step requires native Linux or a VM, not WSL.
4. Flash the WiFi image streamer GAP8 app:

   ```bash
   /Users/sebastianboehler/.local/bin/cfloader flash \
     aideck_gap8_wifi_img_streamer_with_ap.bin \
     deck-bcAI:gap8-fw \
     -w radio://0/80/2M
   ```

5. Connect the laptop to the AI-deck AP named `WiFi streaming example`, or build
   custom Crazyflie firmware to make the AI deck join an existing WiFi network.
6. Run the viewer from the examples checkout:

   ```bash
   git clone https://github.com/bitcraze/aideck-gap8-examples.git
   cd aideck-gap8-examples/examples/other/wifi-img-streamer
   python opencv-viewer.py --save
   ```

The latest `aideck-gap8-examples` release visible during this check was
`2025.02`, and it provides `aideck_gap8_wifi_img_streamer_with_ap.bin`.

## Known Failure Modes

- Camera stream freezes are an active/open issue in
  `bitcraze/aideck-gap8-examples#150`. The issue links a LARICS UDP workaround
  that Bitcraze reports as stable in AP and station mode at about 3 FPS.
- Older image streaming hangs are tracked in
  `bitcraze/aideck-gap8-examples#106`. Comments point to camera-driver hangs,
  power sensitivity, and frequent rebooting as practical workarounds.
- Streaming while flying has been reported to disconnect in
  `bitcraze/crazyflie-firmware#1205`. Bitcraze closed it as network/ESP-specific
  after mixed reproduction, but comments still support testing station mode and
  avoiding VM/network oddities.
- A 2026 report in issue `#150` still reproduces the official TCP freeze in
  station mode with Flow and Multi-ranger attached. The reporter obtained a
  usable stream with the UDP workaround, with occasional JPEG block artifacts.
- The local UDP preparation script pins the LARICS ESP32 firmware, uses the
  current official GAP8 streamer with JPEG plus a 333 ms frame period, and
  provides an explicitly gated radio flash:

  ```bash
  scripts/aideck_udp_streamer.sh build
  AIDECK_UDP_FLASH_CONFIRM=FLASH_AIDECK_UDP \
    scripts/aideck_udp_streamer.sh flash
  ```
- A Bitcraze discussion about standalone streaming suggests motor current draw
  can reset the AI deck. Recommended mitigations were fresh/better battery,
  slower takeoff, and possibly spinning motors before takeoff.
- Exposure behavior on AI deck 1.1 has known quirks. Issue
  `bitcraze/aideck-gap8-examples#63` reports auto exposure not continuously
  adapting; comments mention forcing register updates or extra captures.

## FlightRL Capture Plan

Near-term target:

```text
AI-deck WiFi frames with host timestamps
  + Crazyflie telemetry CSV with host_time_s
  -> aligned multimodal replay dataset
  -> low-res/delta-frame policy experiments in shadow first
```

Do not pipe raw frames through Crazyflie radio logs. Use the AI-deck WiFi
streamer for frames and the existing cflib telemetry logs for state/action
alignment.

The first usable dataset should include:

- frame timestamp
- frame index
- frame path or compressed bytes
- Crazyflie host timestamp
- battery voltage and battery level
- `sys.isFlying`, `sys.isTumbled`, `sys.canfly`
- state estimate position/velocity/attitude
- baseline command and Puffer shadow action

### Vision observation contract

FlightRL exposes a host-fed visual observation contract in
`flightrl.vision`. The stable default is a `64x48` grayscale current frame,
normalized to `[-1, 1]`. Appearance is always retained; temporal information is
optional:

- `frame_stack` adds prior appearance frames;
- `include_delta` adds one signed grayscale difference channel;
- `include_motion_mask` adds one thresholded absolute-change channel;
- `motion_threshold` controls the mask sensitivity;
- `color_mode` can be changed to `rgb` without changing the stream API.

The output layout is channel-first `(C, H, W)` and has a deterministic flattened
view for Puffer/native observation buffers. The planar environment reserves the
vision tail of its observation vector and accepts either raw frames through
`set_vision_frames()` or already processed arrays through
`set_vision_observations()`. The six-DoF simulator does not yet render camera
pixels; use `append_vision_observation()` to compose its state vector with
externally rendered or real frames.

Equivalent task configuration:

```toml
[sensors]
include_vision_sensor = true

[vision]
width = 64
height = 48
color_mode = "grayscale"
input_color_order = "rgb"
frame_stack = 1
include_delta = false
include_motion_mask = false
motion_threshold = 0.08
normalization = "minus_one_one"
```

The representation decision and planned ablations are recorded in
`docs/research/vision_observation_contract_20260724.md`.

Verify the live AI Deck stream through this contract:

```bash
python scripts/capture_aideck_vision.py \
  --transport udp \
  --frames 32 \
  --width 64 \
  --height 48 \
  --output artifacts/ai_deck/vision_observations.npz
```

For a flight, also pass
`--console-output artifacts/ai_deck/<run>-console.jsonl` to the baseline script.
The GAP8 image emits capture, JPEG encode, payload size, and transfer timing at
3 FPS; the NPZ stores `dropped_frames` from UDP resynchronization.

Add temporal channels without changing ingestion:

```bash
python scripts/capture_aideck_vision.py \
  --frames 32 \
  --frame-stack 2 \
  --include-delta \
  --include-motion-mask \
  --motion-threshold 0.08 \
  --output artifacts/ai_deck/vision_observations_temporal.npz
```

Do not fly immediately after changing transport firmware. First require:

1. 300 seconds of ground streaming with radio disconnected.
2. 300 seconds with radio telemetry and console capture active.
3. Repeated viewer stop/start without a power cycle.
4. One propeller-off motor/load diagnostic if the first two pass.

Only then run a hover, followed by slow line/yaw segments. Store the console
stream beside telemetry so `CPX: GAP8` capture/transfer and `CPX: ESP32` WiFi
events identify the failing boundary.

The two 300-second ground gates are automated:

```bash
scripts/aideck_udp_ground_gate.sh stream-only
scripts/aideck_udp_ground_gate.sh radio
```

Set `AIDECK_GATE_FPS` when validating a non-default GAP8 frame rate:

```bash
AIDECK_GATE_DURATION_S=60 AIDECK_GATE_FPS=8 \
  scripts/aideck_udp_ground_gate.sh radio
```

## Measured full-frame limit

The unthrottled QVGA JPEG path was tested on hardware on 2026-07-25 with the
AI Deck 1.1 and Flow Deck v2 mounted:

- short capture: 80 frames at 7.78 FPS, zero dropped frames
- radio coexistence: 480 frames at 7.71 FPS, zero dropped frames
- telemetry: 11,972 samples over 59.94 seconds, 40 ms maximum Crazyflie-time gap
- GAP8 averages: 71.82 ms capture, 57.40 ms JPEG encode, 0.51 ms transfer
- no console errors and no motor output during the ground gate

This is the measured ceiling of the original start-stop capture-then-JPEG
pipeline, not the HM01B0 sensor limit.

The full-QVGA pipelined JPEG variant was subsequently measured with the same
UDP and radio coexistence setup:

- 1,020 frames at 17.12 FPS with zero dropped frames and zero UDP errors
- 18.12 ms mean capture, 57.94 ms mean JPEG encoding, and 0.39 ms transfer
- 11,972 radio telemetry samples with a 38 ms maximum Crazyflie-time gap
- zero motor output, requested motor output, thrust, or flight-state activity

This is a 2.22x sustained improvement over 7.71 FPS without reducing the
`324x244` camera/JPEG representation. JPEG encoding is now the full-resolution
bottleneck.

Build and flash the separate unthrottled image with:

```bash
scripts/aideck_udp_streamer.sh build-max
AIDECK_UDP_FLASH_CONFIRM=FLASH_AIDECK_UDP \
  scripts/aideck_udp_streamer.sh flash-max
```

The normal `flash` command always uses the saved 3 FPS GAP8 artifact as the
rollback image, even after `build-max` has replaced the build-tree output.

## Resolution and encoder benchmark matrix

The GAP8 streamer now exposes build-time controls without changing the
upstream defaults:

- `CAMERA_RESOLUTION=qvga|qqvga`
- `STREAM_ENCODING=raw|jpeg|gray4`
- `STREAM_WIDTH=64` and `STREAM_HEIGHT=48` for packed grayscale
- `JPEG_CLUSTER_OFFLOAD=0|1`
- `OUTPUT_PROFILING_DATA=0|1`
- `TARGET_FRAME_TIME_MS=0|333`

FlightRL defines four unthrottled JPEG benchmark variants:

| Variant | Sensor buffer | JPEG execution |
| --- | --- | --- |
| `qvga-fc` | 324 x 244 | fabric controller |
| `qvga-cluster` | 324 x 244 | GAP8 cluster |
| `qqvga-fc` | 162 x 122 | fabric controller |
| `qqvga-cluster` | 162 x 122 | GAP8 cluster |

The extra two rows and columns match the GAP SDK Himax driver and preserve the
dimensions used by the existing QVGA streamer. Build all variants with:

```bash
scripts/aideck_udp_streamer.sh build-matrix
```

Build or flash one named variant with:

```bash
scripts/aideck_udp_streamer.sh build-benchmark qqvga-cluster
AIDECK_UDP_FLASH_CONFIRM=FLASH_AIDECK_UDP \
  scripts/aideck_udp_streamer.sh flash-benchmark qqvga-cluster
```

All four configurations compile. The generated `qvga-fc` artifact is
byte-identical to the previously measured 7.7 FPS image. The other three
variants require stationary hardware validation before use or an upstream PR.
Test each candidate first with a short frame-integrity capture, then the
stream-only and radio ground gates. Use the direct USB flasher and radio restart
for GAP8 changes, and keep the saved 3 FPS image as the rollback.

## Continuous capture benchmark matrix

The next matrix separates sensor start/stop overhead from JPEG encoding:

| Variant | Capture path | Purpose |
| --- | --- | --- |
| `qqvga-continuous` | one buffer, sensor kept running | Measured 27.46 FPS |
| `qqvga-pipelined` | two buffers, capture overlaps encode/send | Measured 34.31 FPS |
| `qqvga-pipelined-60fps` | two buffers, 60 FPS sensor timing | Measured 55.08 FPS, zero drops |
| `qqvga-pipelined-65fps` | tighter QQVGA timing | Measured 61.72 FPS, two UDP drops |
| `qqvga-pipelined-65fps-gray4` | 64 x 48 packed 4-bit grayscale | Measured 64.83 FPS, zero drops |
| `qqvga-pipelined-100fps-gray4` | 64 x 48 packed grayscale, faster timing | Measured 103.12 FPS, two drops |
| `qqvga-pipelined-120fps-gray4` | minimum QQVGA frame length | Measured 110.95 FPS, one drop |
| `qqvga-pipelined-120fps-gray4-48x36` | one-packet 4-bit grayscale | Measured 111.09 FPS, zero drops |
| `qqvga-pipelined-60fps-raw` | same timing, full raw frame | Transport overload control |
| `qvga-continuous` | one buffer, sensor kept running | QVGA sensor-overhead isolation |
| `qvga-pipelined` | two buffers, capture overlaps encode/send | QVGA encoder-bound ceiling |
| `qvga-pipelined-60fps` | QVGA window and 60 FPS timing | Measured 17.12 FPS, zero drops |

Build all configured capture variants without flashing:

```bash
scripts/aideck_udp_streamer.sh build-capture-matrix
```

All capture modes compile in the official `bitcraze/aideck` container. A clean
checkout of pinned upstream commit `70b8459` plus the saved patch reproduced the
`qqvga-continuous` and 60 FPS JPEG artifacts byte-for-byte. The pipelined QVGA
build retains roughly 183 KiB of L2 headroom after the two frame buffers, JPEG
buffers, and CPX queues.

The default GAP SDK sensor timing uses a 534-line frame. Continuous capture
removes repeated sensor wake/sleep overhead; double buffering then overlaps
capture with JPEG and CPX transfer. The explicit 60 FPS variants enable the
QVGA window, use a 266-line frame, and cap auto-exposure integration at 264
lines. Register writes are read back before streaming.

Hardware validation order is the table order above. Each candidate gets a
short stationary frame-integrity and profiling check before the 60-second radio
coexistence gate. `scripts/flash_aideck_gap8.py` writes GAP8 through USB and
uses the radio power switch to restart the deck, avoiding a manual power cycle.
No flight or motor command is part of these tests.

## Measured compact transport

Full QQVGA raw is not a useful streaming format on the current CPX/ESP32 path.
Its 19,764-byte payload produced only 0.89 FPS, 11 dropped frames out of 12, and
3,138 ESP32 `UDP send failed: errno 12` messages in five seconds. JPEG reduced
the payload to about 1,986 bytes and sustained 61.72 FPS with two UDP failures
over 3,600 frames, but JPEG encoding consumed 15.90 ms per frame.

The `gray4` path performs the policy-oriented preprocessing on GAP8:

```text
162x122 sensor grayscale
  -> nearest-neighbor 64x48
  -> 4-bit quantization
  -> two pixels per byte
  -> 1,536-byte CPX/UDP payload
  -> receiver expands nibbles to uint8 grayscale
```

The 60-second radio coexistence gate produced:

- 3,840 frames at 64.83 FPS with zero dropped frames
- zero ESP32 UDP failures
- 0.36 ms mean resize/pack time and 0.13 ms mean CPX transfer time
- 11,972 telemetry samples with a 34 ms maximum Crazyflie-time gap
- zero motor output, requested motor output, thrust, or flight-state activity

This is the preferred high-rate policy transport. It does not replace
occasional JPEG or full-quality captures for training data. The faster sensor
timing also limits exposure, so image brightness and 4-bit policy accuracy must
be tested as separate ablations before visual control.

The same representation reached 103.12 FPS at the 100 FPS timing candidate and
110.95 FPS at the 120 FPS candidate. Those 60-second gates produced five and
one ESP32 `errno 12` send failures respectively, so neither replaces the
zero-drop 64.83 FPS baseline. The 65 FPS gray4 image was restored and verified
at 64.75 FPS after the ceiling tests.

A `48x36` gray4 variant produces an 864-byte payload, which fits in one
1,020-byte GAP8 CPX application packet. It sustained 111.09 FPS for 6,600
frames with zero drops and zero UDP errors. This validates packet count as the
high-rate transport constraint, but the lower spatial resolution remains an
optional benchmark. The `64x48` zero-drop policy image was restored after the
test.

On 2026-07-26 the `qqvga-pipelined-65fps-gray4` policy image was flashed again
and left active on the AI Deck. Its SHA-256 is
`da8fef120813f162d144a969a24e08ae8e7843b855ae2710478b7556b91ab08d`.
A post-flash concurrent gate recorded 650 `64x48` observations at 64.86 FPS
with zero drops and 499 radio samples with a 20 ms maximum timestamp gap. All
four motor outputs and `sys.isFlying` remained zero. The vision artifact is in
`artifacts/ai_deck/policy_image_active_gate_20260726T0703Z/vision.npz`.

## Switching to host semantic grounding

The active `64x48` gray4 profile is too small and dark for the first
open-vocabulary Grounding DINO test. A named command restores the existing
`162x122` pipelined JPEG artifact without rebuilding it:

```bash
AIDECK_UDP_FLASH_CONFIRM=FLASH_AIDECK_UDP \
  scripts/aideck_udp_streamer.sh flash-semantic
```

Power cycle the Crazyflie after flashing, connect the Mac to the AI Deck Wi-Fi
access point, and run the non-flying gate:

```bash
uv run --extra semantic --extra hardware \
  python scripts/crazyflie_semantic_find.py \
  --prompt "computer monitor" \
  --duration-s 10
```

The runner rejects `64x48` input and dark frames before any radio connection or
motor command. Each accepted run writes its manifest, raw and annotated frames,
grounding events, and summary below `artifacts/semantic/`.

The next optimization order is:

1. Add measured ESP32 send backpressure and allocation-failure counters.
2. Test a sub-1 KB packed tensor that fits one CPX payload.
3. Compare current gray4 against GAP8-side delta, motion, or learned features.
4. Raise sensor timing only after the selected representation has a zero-drop
   transport gate.

## Upstream contribution boundary

An upstream contribution is justified by Bitcraze issue
`aideck-gap8-examples#137`, which asks where the FPS bottleneck is and how to
improve it stably. Do not submit the current experiment as one large PR.

Recommended split:

1. A small correctness PR for final CPX chunk copying and the redundant
   zero-length packet.
2. A full-resolution performance PR for continuous capture, double buffering,
   capture/encode overlap, timing register readback, and profiling. Lead with
   the QVGA result: 7.71 to 17.12 FPS with unchanged image representation.
3. A separate optional packed-image PR only after its wire format and bit depth
   are generalized and documented.

The current build controls already keep resolution and encoding optional:
`CAMERA_RESOLUTION`, `STREAM_ENCODING`, `STREAM_WIDTH`, and `STREAM_HEIGHT`.
Upstream defaults remain QVGA, raw, and start-stop capture. The packed format
still fixes quantization at four bits, so it is not ready to be the upstream
default or the headline PR.

## Model Implications

The AI deck should not automatically improve low-level flight stability. Flow
deck plus firmware estimator already handle near-floor stabilization. Camera
data helps with observability:

- object following
- visual target alignment
- obstacle/room context beyond point range sensors
- low-bandwidth pixel or frame-difference policies
- learned perception features that can later run on host or GAP8

For stability, the useful path is indirect: use vision to estimate drift,
target position, obstacle proximity, or scene context, then feed a bounded
setpoint policy. Keep the stabilizer and safety gates in charge of low-level
flight.

Recommended first policy lane:

1. Record synchronized frames plus telemetry during baseline-controlled shadow
   flights.
2. Downsample to grayscale patches or frame differences.
3. Train a small recurrent encoder plus the existing state/flow policy head.
4. Replay-gate against held-out logs.
5. Shadow-run live.
6. Only then consider bounded visual setpoint control.

## 2026-07-07 Recovery Notes

After a user power cycle, scan temporarily saw both:

```text
radio://0/0/250K
radio://0/80/2M
```

`cfloader info -w radio://0/80/2M/E7E7E7E7E7` succeeded and reported the
STM32 and nRF51 bootloaders. This proves the base radio and bootloader path
were alive at that point.

`cfloader reset` and `cfloader reset -w ...` failed in the local cfloader
wrapper with:

```text
AttributeError: 'NoneType' object has no attribute 'send_packet'
```

Direct `cflib.bootloader.cloader.Cloader` reset probes then found no responding
bootloader on:

```text
radio://0/0/250K
radio://0/0/2M/E7E7E7E7E7
radio://0/110/2M/E7E7E7E7E7
radio://0/80/2M/E7E7E7E7E7
radio://0/80/2M
```

Subsequent scans reported no Crazyflie interfaces, while macOS still detected
the Crazyradio 2.0 USB dongle (`1915:7777 Bitcraze AB Crazyradio 2.0`). This
means the immediate recovery blocker is Crazyflie radio/boot state, not host USB
radio enumeration.

Next recovery step: fully remove power from the Crazyflie, including USB if
attached, wait a few seconds, then power it from battery and verify LED activity
before another scan. Do not start another AI-deck GAP8 flash until a normal
firmware API connection and deck parameter read are stable again.

## 2026-07-07 Base Firmware Recovery Attempt

After a full battery-only reboot, scan again reported:

```text
radio://0/80/2M
```

Normal `cflib.crazyflie.Crazyflie.open_link()` still opened a radio link object
but never emitted `connected` or `fully_connected` within 20-25 seconds, for
both:

```text
radio://0/80/2M/E7E7E7E7E7
radio://0/80/2M
```

A base-firmware-only recovery flash was then run:

```text
cfloader flash artifacts/firmware/firmware-cf2-2026.04.zip stm32-fw nrf51-fw -w radio://0/80/2M/E7E7E7E7E7
```

It wrote both targets successfully:

```text
Flashing 1 of 2 to stm32 (fw): 290795 bytes (284 pages)
Flashing 2 of 2 to nrf51 (fw): 47019 bytes (46 pages)
```

Because the CLI reset path fails locally, a direct `cflib.bootloader.Bootloader`
reset was used:

```text
start_bootloader True protocol 16
reset_to_firmware True
```

After reset-to-firmware, scan returned to `radio://0/80/2M`, but the firmware
API still did not emit `connected` or `fully_connected`. Do not fly in this
state. The next escalation should be a Bitcraze Discussion or issue with the
base firmware recovery log attached.

## 2026-07-07 AI-Deck-Only Retry

After checking Bitcraze discussion #1694, the Flow deck was physically removed
and the drone was tested with only the AI Deck attached. The base firmware
recovery flash again succeeded:

```text
Flashing 1 of 2 to stm32 (fw): 290795 bytes (284 pages)
Flashing 2 of 2 to nrf51 (fw): 47019 bytes (46 pages)
```

Direct bootloader reset again succeeded:

```text
start_bootloader True protocol 16
reset_to_firmware True
```

Scan then returned:

```text
radio://0/80/2M
```

But `cflib.crazyflie.Crazyflie.open_link()` still did not emit `connected` or
`fully_connected` within 30 seconds. This means removing the Flow deck did not
restore the normal firmware API. Keep the aircraft grounded and escalate with
the AI-deck-only recovery log:

```text
artifacts/firmware/flash_base_cf2_2026_04_ai_deck_only_20260707.log
```

## 2026-07-07 USB Diagnostic Attempt

The drone was power-cycled again and connected over micro USB for a final
diagnostic. macOS and `pyusb` did not enumerate a Crazyflie USB CRTP device; only
the Crazyradio 2.0 was visible:

```text
1915:7777 Bitcraze AB Crazyradio 2.0
```

`cflib` expects Crazyflie USB CRTP as VID/PID `0483:5740` with manufacturer
`Bitcraze AB`. Direct USB probing failed:

```text
CfUsb scan: []
connection_failed usb://0 Could not open usb://0
```

Radio scan initially still returned `radio://0/80/2M`, and one more
`cfloader info -w radio://0/80/2M/E7E7E7E7E7` reached the bootloader. After that,
direct reset-to-firmware fallback probes could not reattach on the tested radio
URI variants, and the final scan returned no Crazyflie interfaces. The next
practical step is Bitcraze support/discussion or a hardware recovery path such
as verified USB/DFU/JTAG guidance from Bitcraze.
