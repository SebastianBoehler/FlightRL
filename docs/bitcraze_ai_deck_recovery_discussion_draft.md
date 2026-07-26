# Bitcraze Discussion Draft

## Suggested title

Crazyflie 2.1 Brushless + AI Deck 1.1: radio scan works after firmware recovery, but firmware API never connects

## Suggested category

Q&A

## Draft body

Hi Bitcraze team,

I am trying to recover a Crazyflie 2.1 Brushless with an AI Deck 1.1 attached.
The bootloader path over Crazyradio works and base firmware flashing succeeds,
but the normal firmware API never reaches `connected` / `fully_connected` in
`cflib`.

I am looking for guidance on the right next recovery step: AI-deck/GAP8
recovery, deck/contact isolation, Crazyflie firmware recovery, USB/DFU, or a
client-side issue.

### Hardware / software

- Crazyflie 2.1 Brushless
- Crazyradio 2.0
- AI Deck 1.1 mounted on top
- Flow deck v2 was initially mounted on bottom, then removed for an AI-deck-only
  retry after reading discussion #1694.
- Multi-ranger/ranger deck removed
- macOS host
- `cflib`: 0.1.32
- Crazyflie release zip: `firmware-cf2-2026.04.zip`
- AI-deck GAP8 WiFi streamer binary:
  `aideck_gap8_wifi_img_streamer_with_ap_2025.02.bin`

I have not yet tried the VCOM-to-GND resistor workaround or JTAG GAP8 recovery.

### Relevant context

I checked the AI-deck getting started guide and discussions #1400, #1694, and
#2018. In #1694, the recent maintainer reply says the old 3% GAP8 flashing issue
should have been addressed by a conditional boot delay, and asks for a new
discussion if this still happens on recent firmware.

### Symptoms / recovery attempts

- The AI deck was detected at least once after reseating a moved pin/contact:

```text
deck.bcAI=1
deck.bcFlow2=1
deck.bcZRanger2=1
```

- GAP8 WiFi streamer flashing over radio repeatedly stalled before normal GAP8
  write progress:

```text
cfloader flash aideck_gap8_wifi_img_streamer_with_ap_2025.02.bin deck-bcAI:gap8-fw -w radio://0/80/2M/E7E7E7E7E7
Reset to bootloader mode ...
Could not save cache, no writable directory
Could not save cache, no writable directory
```

It never reached the expected `Deck bcAI:gap8` write-progress output.

- Flashing the AI-deck ESP target from the 2026.04 firmware zip did succeed.

- Bootloader info over radio works:

```text
cfloader info -w radio://0/80/2M/E7E7E7E7E7
Connected to bootloader on Crazyflie 2.0 (version=0x10)
Target info: stm32 (0xFF) | Version: None
Target info: nrf51 (0xFE) | Version: 2024.10.0
```

- Base firmware recovery also succeeds, including after removing the Flow deck
  and retrying with only the AI Deck attached:

```text
cfloader flash firmware-cf2-2026.04.zip stm32-fw nrf51-fw -w radio://0/80/2M/E7E7E7E7E7
Flashing 1 of 2 to stm32 (fw): 290795 bytes (284 pages)
Flashing 2 of 2 to nrf51 (fw): 47019 bytes (46 pages)
```

- The local `cfloader reset` command fails with:

```text
AttributeError: 'NoneType' object has no attribute 'send_packet'
```

Using `cflib.bootloader.Bootloader` directly did work at least once:

```text
start_bootloader True protocol 16
reset_to_firmware True
```

- After reset, radio scan returns `radio://0/80/2M`, but normal firmware
  connection still does not complete:

```text
opening radio://0/80/2M/E7E7E7E7E7
wait 1s
...
wait 30s
state {'connected': False, 'fully': False} link open? True
```

The same happens with the shorter scanned URI `radio://0/80/2M`.

- A raw CRTP platform-version request did not receive a platform response. The
  link produced repeated empty port 15/channel 3 packets, but no platform
  packet.

- As a final diagnostic, I power-cycled and attached micro USB. macOS/pyusb did
  not enumerate a Crazyflie USB CRTP device (`0483:5740` / `Bitcraze AB`). Only
  the Crazyradio 2.0 was visible:

```text
1915:7777 Bitcraze AB Crazyradio 2.0
CfUsb scan: []
connection_failed usb://0 Could not open usb://0
```

### Questions

1. Given that bootloader info and base firmware flashing work, but the firmware
   API does not answer, what should I try next?
2. Should I remove the AI Deck too and test with only the Crazyflie base board?
3. Does this look like an AI Deck 1.1 / GAP8 bootloader issue even though the
   immediate failure is now the Crazyflie firmware API?
4. Is JTAG GAP8 bootloader recovery the recommended next step, or should I first
   recover/verify the Crazyflie through USB DFU or cfclient?
5. Is the `cfloader reset` `NoneType.send_packet` traceback a known
   crazyflie-clients-python issue?
6. Should Crazyflie 2.1 Brushless enumerate as `0483:5740` for `usb://0` CRTP
   in normal firmware mode, or is that not expected in this board/state?

I can attach full logs if useful.
