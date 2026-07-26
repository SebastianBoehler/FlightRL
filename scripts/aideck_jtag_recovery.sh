#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOOTLOADER_DIR="$ROOT/external/aideck-gap8-bootloader"
OPENOCD_DIR="$ROOT/external/gap8-openocd"
OPENOCD_BIN="$OPENOCD_DIR/build-native/bin/gap8-openocd"
OPENOCD_SCRIPTS="$OPENOCD_DIR/build-native/share/openocd/scripts"
OPENOCD_TOOLS="$ROOT/external/gap8-openocd-tools"
FLASH_IMAGE="$BOOTLOADER_DIR/BUILD/GAP8_V2/GCC_RISCV_FREERTOS/target.board.devices.flash.img"
STATE_DIR="$ROOT/artifacts/ai_deck/jtag_recovery"

BOOTLOADER_COMMIT="3775bc3eca232b5f4c4fe952ccbc6bf8005dd2e5"
OPENOCD_COMMIT="b84d97ec4d2e601e704b54351e954b1c58d41683"
FLASH_IMAGE_SHA256="1b0a299f2cdf6649c91cb38e2b34655b4461ebee5c57b475505975ab5227dc9b"
FLASHER_SHA256="e07cc2461b8e8e691e9cd7ffd4d9e50a2043bb4301ebabd3dea07641a07b6e47"
FLASH_IMAGE_SIZE=76832

usage() {
  cat <<'EOF'
Usage: scripts/aideck_jtag_recovery.sh COMMAND

Commands:
  preflight  Validate the pinned recovery inputs and native OpenOCD build
  status     Show whether macOS detects the Olimex programmer
  probe      Read the GAP8 JTAG TAP ID without writing to the deck
  flash      Restore the GAP8 bootloader, then run a read-only probe

The flash command requires:
  AIDECK_FLASH_CONFIRM=FLASH_GAP8_BOOTLOADER
EOF
}

require_command() {
  command -v "$1" >/dev/null || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

sha256() {
  shasum -a 256 "$1" | awk '{print $1}'
}

mac_has_olimex() {
  ioreg -p IOUSB -l -w 0 2>/dev/null |
    grep -q 'Olimex OpenOCD JTAG ARM-USB-TINY-H'
}

validate_static_inputs() {
  require_command file
  require_command git
  require_command ioreg
  require_command shasum

  test -x "$OPENOCD_BIN" || {
    echo "Missing native GAP8 OpenOCD binary: $OPENOCD_BIN" >&2
    exit 1
  }
  file "$OPENOCD_BIN" | grep -q 'Mach-O 64-bit executable arm64' || {
    echo "GAP8 OpenOCD is not a native Apple Silicon executable." >&2
    exit 1
  }

  local commit
  commit="$(git -C "$BOOTLOADER_DIR" rev-parse HEAD)"
  test "$commit" = "$BOOTLOADER_COMMIT" || {
    echo "Unexpected bootloader commit: $commit" >&2
    exit 1
  }

  commit="$(git -C "$OPENOCD_DIR" rev-parse HEAD)"
  test "$commit" = "$OPENOCD_COMMIT" || {
    echo "Unexpected GAP8 OpenOCD commit: $commit" >&2
    exit 1
  }

  test "$(sha256 "$FLASH_IMAGE")" = "$FLASH_IMAGE_SHA256" || {
    echo "Bootloader flash image hash mismatch." >&2
    exit 1
  }
  test "$(stat -f '%z' "$FLASH_IMAGE")" -eq "$FLASH_IMAGE_SIZE" || {
    echo "Bootloader flash image size mismatch." >&2
    exit 1
  }

  local flasher="$OPENOCD_TOOLS/gap_bins/gap_flasher-gapoc_a.elf"
  test "$(sha256 "$flasher")" = "$FLASHER_SHA256" || {
    echo "GAP8 flasher hash mismatch." >&2
    exit 1
  }
  test -f "$OPENOCD_TOOLS/tcl/flash_image.tcl"
  test -f "$OPENOCD_TOOLS/tcl/jtag_boot.tcl"

  mkdir -p "$STATE_DIR"
  echo "Recovery preflight passed."
  echo "Bootloader: $BOOTLOADER_COMMIT"
  echo "OpenOCD:   $OPENOCD_COMMIT (native arm64)"
  echo "Image:     $FLASH_IMAGE_SHA256"
}

require_programmer() {
  mac_has_olimex || {
    echo "Olimex ARM-USB-TINY-H (15ba:002a) is not visible on macOS." >&2
    exit 1
  }
}

probe_gap8() {
  validate_static_inputs
  require_programmer

  local timestamp log
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  log="$STATE_DIR/probe-gap8-$timestamp.log"

  set +e
  "$OPENOCD_BIN" \
    -s "$OPENOCD_SCRIPTS" \
    -d2 \
    -c 'gdb_port disabled; telnet_port disabled; tcl_port disabled' \
    -c 'script interface/ftdi/olimex-arm-usb-tiny-h.cfg; script target/gap8revb.tcl; tap_select gap8_adv_debug_itf; du_select adv_dbg_unit 7; init; scan_chain; targets; shutdown' \
    2>&1 | tee "$log"
  local status=${PIPESTATUS[0]}
  set -e

  test "$status" -eq 0 || {
    echo "GAP8 JTAG probe failed. Log: $log" >&2
    exit "$status"
  }
  grep -qi 'tap/device found: 0x149511c3' "$log" || {
    echo "Probe exited without the expected GAP8 TAP ID. Log: $log" >&2
    exit 1
  }
  echo "GAP8 JTAG probe passed. Log: $log"
}

flash_bootloader() {
  test "${AIDECK_FLASH_CONFIRM:-}" = "FLASH_GAP8_BOOTLOADER" || {
    echo "Refusing to flash without AIDECK_FLASH_CONFIRM=FLASH_GAP8_BOOTLOADER" >&2
    exit 1
  }
  validate_static_inputs
  require_programmer

  local timestamp log openocd_commands
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  log="$STATE_DIR/flash-gap8-bootloader-$timestamp.log"
  openocd_commands="script interface/ftdi/olimex-arm-usb-tiny-h.cfg; "
  openocd_commands+="script target/gap8revb.tcl; "
  openocd_commands+="script $OPENOCD_TOOLS/tcl/flash_image.tcl; "
  openocd_commands+="script $OPENOCD_TOOLS/tcl/jtag_boot.tcl; "
  openocd_commands+="gap_flash_raw_hyper $FLASH_IMAGE $FLASH_IMAGE_SIZE $OPENOCD_TOOLS; exit;"

  set +e
  "$OPENOCD_BIN" \
    -s "$OPENOCD_SCRIPTS" \
    -d2 \
    -c 'gdb_port disabled; telnet_port disabled; tcl_port disabled' \
    -c "$openocd_commands" \
    2>&1 | tee "$log"
  local status=${PIPESTATUS[0]}
  set -e

  test "$status" -eq 0 || {
    echo "GAP8 bootloader flash failed. Log: $log" >&2
    exit "$status"
  }
  grep -q 'copied 76832 / 76832 Bytes' "$log" || {
    echo "Flash did not report a complete image transfer. Log: $log" >&2
    exit 1
  }
  grep -q 'flasher is done!' "$log" || {
    echo "Flash exited without Bitcraze's success marker. Log: $log" >&2
    exit 1
  }

  echo "GAP8 bootloader flash completed. Log: $log"
  probe_gap8
}

case "${1:-}" in
  preflight)
    validate_static_inputs
    ;;
  status)
    echo "macOS Olimex: $(mac_has_olimex && echo visible || echo not-visible)"
    ;;
  probe)
    probe_gap8
    ;;
  flash)
    flash_bootloader
    ;;
  *)
    usage
    exit 2
    ;;
esac
