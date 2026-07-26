# AI Deck GAP8 JTAG Recovery

Checked against Bitcraze documentation on 2026-07-24.

## Recovery Result

Recovery completed on 2026-07-24. Native OpenOCD identified GAP8 TAP
`0x149511c3`, the JTAG flasher copied all 76,832 bootloader bytes, and the
post-flash probe passed. After a power cycle, `deck.bcAI=1` was detected over
Crazyradio. The official `2025.02` WiFi streamer subsequently reached 100%,
advertised `WiFi streaming example`, and produced a valid 324x244 camera frame.

## Why This Recovery Is Required

The official WiFi streamer reached 99% in the previous direct deck-memory
attempt, then timed out without a completion acknowledgement. This is not a
validated GAP8 application flash. Bitcraze recommends restoring the GAP8
bootloader over JTAG when OTA GAP8 flashing does not complete.

## Prepared Inputs

- `external/aideck-gap8-bootloader` at Bitcraze commit
  `3775bc3eca232b5f4c4fe952ccbc6bf8005dd2e5`
- native Apple Silicon build of GreenWaves GAP8 OpenOCD at commit
  `b84d97ec4d2e601e704b54351e954b1c58d41683`
- GAP8 flash helpers extracted from the pinned Bitcraze AI Deck image
- prebuilt GAP8 bootloader flash image, SHA-256
  `1b0a299f2cdf6649c91cb38e2b34655b4461ebee5c57b475505975ab5227dc9b`
- official WiFi streamer `2025.02`, SHA-256
  `13348d9303fc39baef0d99c269afa2f7c692f260f2c3b937cbf649aea6888034`
- official examples checkout at tag `2025.02`
- Crazyflie 2.1 Brushless release `firmware-cf21bl-2026.04.zip`, SHA-256
  `ca8743ce882217b4a09c711b12549e2d70fbc92aa576166317129beca6aaa64c`

Do not use `firmware-cf2-2026.04.zip` for this aircraft. The correct platform
bundle is `cf21bl`.

## Physical Setup

1. Remove the propellers.
2. Power off the Crazyflie and disconnect its battery and Micro-USB.
3. Leave only the AI Deck mounted. Remove the Flow Deck and other decks.
4. Remove the wide grey 20-pin extension cable from the ARM-USB-TINY-H.
5. Plug the ARM-JTAG-20-10 adapter directly into the programmer's 20-pin
   header.
6. Connect the adapter's small 10-pin ribbon to the AI Deck GAP8 JTAG header.
   Use the left header when viewing the deck from above with the camera at the
   front. Match the ribbon's pin-1 edge to the deck's pin-1 marking.
7. Connect the programmer to the Mac with the USB-C-to-USB-B data cable.
8. Reconnect the Crazyflie battery and power it on only after all JTAG
   connectors are seated.

Do not use the AI Deck ESP32 JTAG header. Do not force a connector or move one
side by one pin.

## Operator Commands

Software-only validation:

```bash
scripts/aideck_jtag_recovery.sh preflight
```

After the programmer and powered target are connected:

```bash
scripts/aideck_jtag_recovery.sh status
scripts/aideck_jtag_recovery.sh probe
```

The flash remains deliberately blocked until explicitly confirmed:

```bash
AIDECK_FLASH_CONFIRM=FLASH_GAP8_BOOTLOADER \
  scripts/aideck_jtag_recovery.sh flash
```

Logs are written to `artifacts/ai_deck/jtag_recovery/`.

## Required Success Evidence

The read-only probe must identify TAP ID `0x149511c3`. The flash is successful
only when the log reports `copied 76832 / 76832 Bytes`, contains Bitcraze's
`flasher is done!` marker, and the automatic post-flash probe succeeds. A
partial percentage, process timeout, radio reconnect, or programmer LED alone
is not success.

After a successful bootloader restore:

1. Disconnect JTAG and fully power-cycle the Crazyflie.
2. Verify normal firmware connection and `deck.bcAI=1`.
3. Flash the official WiFi streamer over Crazyradio.
4. Require 100% completion.
5. Verify the `WiFi streaming example` access point and capture repeatable
   camera frames before remounting the Flow Deck.

The isolated viewer environment is already prepared:

```bash
artifacts/ai_deck/viewer-venv/bin/python \
  external/aideck-gap8-examples/examples/other/wifi-img-streamer/opencv-viewer.py
```

It contains OpenCV separately from `cfclient`, as recommended by Bitcraze.

Bitcraze currently warns that the GreenWaves AutoTiler download is unavailable.
That does not block bootloader restoration or the prebuilt WiFi streamer, but
it does block new GAP SDK neural-network deployment unless the licensed
AutoTiler is already available. Bitcraze points to DORY as the current
alternative.

## Sources

- https://www.bitcraze.io/documentation/tutorials/getting-started-with-aideck/
- https://www.bitcraze.io/documentation/repository/aideck-gap8-examples/master/development/jtag-programmer/
- https://github.com/bitcraze/aideck-gap8-bootloader
