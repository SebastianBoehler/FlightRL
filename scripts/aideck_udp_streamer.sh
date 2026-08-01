#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ESP_DIR="$ROOT/external/aideck-esp-firmware-udp"
GAP8_DIR="$ROOT/external/aideck-gap8-examples"
PATCH="$ROOT/firmware/aideck_udp/wifi_streamer_udp.patch"
ARTIFACT_DIR="$ROOT/artifacts/ai_deck/udp_firmware"

ESP_REPO="https://github.com/larics/aideck-esp-firmware-udp.git"
GAP8_REPO="https://github.com/bitcraze/aideck-gap8-examples.git"
ESP_COMMIT="2b7152366048bbfa98eea343f68264c52ba0bd0d"
GAP8_COMMIT="70b84590baa0a5fa7b98ea98842ed407de4dabd6"
CFLOADER="${CFLOADER:-cfloader}"
AIDECK_URI="${AIDECK_URI:-radio://0/80/2M/E7E7E7E7E7}"
AIDECK_GAP8_URI="${AIDECK_GAP8_URI:-usb://0}"
GAP8_FLASHER="$ROOT/scripts/flash_aideck_gap8.py"

ESP_BOOTLOADER="$ESP_DIR/build/bootloader/bootloader.bin"
ESP_IMAGE="$ESP_DIR/build/aideck_esp.bin"
GAP8_IMAGE="$GAP8_DIR/examples/other/wifi-img-streamer/BUILD/GAP8_V2/GCC_RISCV_FREERTOS/target.board.devices.flash.img"
GAP8_STABLE_IMAGE="$ARTIFACT_DIR/aideck-gap8-udp-jpeg-3fps.img"
GAP8_MAX_IMAGE="$ARTIFACT_DIR/aideck-gap8-udp-jpeg-maxfps.img"
GAP8_SEMANTIC_SAFE_IMAGE="$ARTIFACT_DIR/aideck-gap8-qvga-jpeg-pipelined-60fps-frame-safe.img"
GAP8_POLICY_SAFE_IMAGE="$ARTIFACT_DIR/aideck-gap8-qqvga-gray4-pipelined-65fps-frame-safe.img"
GAP8_BENCHMARK_VARIANTS=(qvga-fc qvga-cluster qqvga-fc qqvga-cluster)
GAP8_CAPTURE_VARIANTS=(
  qvga-continuous
  qvga-pipelined
  qvga-pipelined-60fps
  qqvga-continuous
  qqvga-pipelined
  qqvga-pipelined-60fps
  qqvga-pipelined-65fps
  qqvga-pipelined-65fps-gray4
  qqvga-pipelined-100fps-gray4
  qqvga-pipelined-120fps-gray4
  qqvga-pipelined-120fps-gray4-48x36
  qqvga-pipelined-60fps-raw
)

usage() {
  cat <<'EOF'
Usage: scripts/aideck_udp_streamer.sh COMMAND

Commands:
  prepare    Fetch pinned sources and apply the UDP/JPEG streamer patch
  preflight Validate source revisions, patch, Docker, and cfloader
  build      Build the ESP32 UDP firmware and 3 FPS JPEG GAP8 app
  build-max  Build an unthrottled JPEG GAP8 app for ground testing
  build-benchmark VARIANT
             Build qvga-fc, qvga-cluster, qqvga-fc, or qqvga-cluster
  build-matrix
             Build all four unthrottled GAP8 benchmark variants
  build-capture-matrix
             Build continuous, pipelined, fast-sensor, raw, and gray4 variants
  status     Print source and image status
  flash      Flash GAP8 and ESP32 images over Crazyradio
  flash-max  Flash only the unthrottled GAP8 image over Crazyradio
  flash-benchmark VARIANT
             Flash one previously built GAP8 benchmark variant
  flash-semantic
             Flash the frame-safe 324x244 JPEG profile used for text grounding
  flash-semantic-highres
             Alias for the frame-safe semantic profile
  flash-policy-safe
             Flash the frame-safe 64x48 packed gray4 policy profile

The flash command requires:
  AIDECK_UDP_FLASH_CONFIRM=FLASH_AIDECK_UDP

Override the default radio URI with AIDECK_URI.
EOF
}

require_command() {
  command -v "$1" >/dev/null || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

clone_pinned() {
  local repo="$1"
  local path="$2"
  local commit="$3"
  if test ! -d "$path/.git"; then
    git clone "$repo" "$path"
    git -C "$path" checkout --detach "$commit"
  fi
  test "$(git -C "$path" rev-parse HEAD)" = "$commit" || {
    echo "Unexpected source revision in $path" >&2
    exit 1
  }
}

prepare_sources() {
  require_command git
  clone_pinned "$ESP_REPO" "$ESP_DIR" "$ESP_COMMIT"
  clone_pinned "$GAP8_REPO" "$GAP8_DIR" "$GAP8_COMMIT"

  if git -C "$GAP8_DIR" apply --check "$PATCH" 2>/dev/null; then
    git -C "$GAP8_DIR" apply "$PATCH"
  elif ! git -C "$GAP8_DIR" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "GAP8 UDP patch is neither applicable nor already applied." >&2
    exit 1
  fi
}

validate_sources() {
  prepare_sources
  require_command docker
  require_command "$CFLOADER"
  test -s "$GAP8_FLASHER" || {
    echo "GAP8 flasher is missing: $GAP8_FLASHER" >&2
    exit 1
  }
  grep -q 'STREAM_ENCODING_MODE' \
    "$GAP8_DIR/examples/other/wifi-img-streamer/wifi-img-streamer.c"
  grep -q 'CAMERA_RESOLUTION' \
    "$GAP8_DIR/examples/other/wifi-img-streamer/Makefile"
  grep -q 'CAPTURE_MODE' \
    "$GAP8_DIR/examples/other/wifi-img-streamer/Makefile"
  grep -q 'SENSOR_FRAME_RATE' \
    "$GAP8_DIR/examples/other/wifi-img-streamer/Makefile"
  test -s \
    "$GAP8_DIR/examples/other/wifi-img-streamer/himax_timing.c"
  grep -q 'SOCK_DGRAM' "$ESP_DIR/main/wifi.c"
  docker info >/dev/null
  echo "AI Deck UDP preflight passed."
}

build_images() {
  validate_sources
  docker run --rm \
    -v "$ESP_DIR:/module" \
    -w /module \
    bitcraze/builder \
    /bin/bash -lc \
    'source /new_home/.espressif/python_env/idf4.3_py3.10_env/bin/activate && make -j2'

  docker run --rm \
    -v "$GAP8_DIR:/module" \
    bitcraze/aideck \
    tools/build/make-example \
    examples/other/wifi-img-streamer \
    clean build image \
    SETUP_WIFI_AP=1 \
    CAMERA_RESOLUTION=qvga \
    CAPTURE_MODE=start-stop \
    STREAM_ENCODING=jpeg \
    JPEG_CLUSTER_OFFLOAD=0 \
    OUTPUT_PROFILING_DATA=1 \
    TARGET_FRAME_TIME_MS=333

  test -s "$ESP_BOOTLOADER"
  test -s "$ESP_IMAGE"
  test -s "$GAP8_IMAGE"
  mkdir -p "$ARTIFACT_DIR"
  cp "$ESP_BOOTLOADER" "$ARTIFACT_DIR/esp32-bootloader.bin"
  cp "$ESP_IMAGE" "$ARTIFACT_DIR/aideck-esp-udp.bin"
  cp "$GAP8_IMAGE" "$GAP8_STABLE_IMAGE"
  (
    cd "$ARTIFACT_DIR"
    shasum -a 256 ./*.bin ./*.img > SHA256SUMS
  )
  {
    echo "built_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "esp_commit=$ESP_COMMIT"
    echo "gap8_commit=$GAP8_COMMIT"
    echo "gap8_patch_sha256=$(shasum -a 256 "$PATCH" | awk '{print $1}')"
    echo "transport=udp"
    echo "encoding=jpeg"
    echo "target_fps=3"
  } > "$ARTIFACT_DIR/BUILD_INFO.txt"
  echo "Built UDP firmware in $ARTIFACT_DIR"
}

build_max_image() {
  build_benchmark_variant qvga-fc
  cp "$BENCHMARK_IMAGE" "$GAP8_MAX_IMAGE"
  shasum -a 256 "$GAP8_MAX_IMAGE" > "$GAP8_MAX_IMAGE.sha256"
  echo "Built unthrottled GAP8 image: $GAP8_MAX_IMAGE"
}

flash_gap8_image() {
  uv run --extra hardware python "$GAP8_FLASHER" \
    --image "$1" \
    --uri "$AIDECK_GAP8_URI" \
    --reboot-uri "$AIDECK_URI" \
    --confirm FLASH_AIDECK_GAP8
}

set_benchmark_variant() {
  BENCHMARK_ENCODING=jpeg
  BENCHMARK_SENSOR_FRAME_RATE=default
  BENCHMARK_STREAM_WIDTH=64
  BENCHMARK_STREAM_HEIGHT=48
  case "$1" in
    qvga-fc)
      BENCHMARK_RESOLUTION=qvga
      BENCHMARK_CAPTURE_MODE=start-stop
      BENCHMARK_CLUSTER_OFFLOAD=0
      BENCHMARK_ENCODER=fc
      ;;
    qvga-cluster)
      BENCHMARK_RESOLUTION=qvga
      BENCHMARK_CAPTURE_MODE=start-stop
      BENCHMARK_CLUSTER_OFFLOAD=1
      BENCHMARK_ENCODER=cluster
      ;;
    qqvga-fc)
      BENCHMARK_RESOLUTION=qqvga
      BENCHMARK_CAPTURE_MODE=start-stop
      BENCHMARK_CLUSTER_OFFLOAD=0
      BENCHMARK_ENCODER=fc
      ;;
    qqvga-cluster)
      BENCHMARK_RESOLUTION=qqvga
      BENCHMARK_CAPTURE_MODE=start-stop
      BENCHMARK_CLUSTER_OFFLOAD=1
      BENCHMARK_ENCODER=cluster
      ;;
    qvga-continuous|qvga-pipelined|qqvga-continuous|qqvga-pipelined)
      BENCHMARK_RESOLUTION="${1%%-*}"
      BENCHMARK_CAPTURE_MODE="${1#*-}"
      BENCHMARK_CLUSTER_OFFLOAD=0
      BENCHMARK_ENCODER=fc
      ;;
    qvga-pipelined-60fps|qqvga-pipelined-60fps)
      BENCHMARK_RESOLUTION="${1%%-*}"
      BENCHMARK_CAPTURE_MODE=pipelined
      BENCHMARK_SENSOR_FRAME_RATE=60
      BENCHMARK_CLUSTER_OFFLOAD=0
      BENCHMARK_ENCODER=fc
      ;;
    qqvga-pipelined-65fps)
      BENCHMARK_RESOLUTION=qqvga
      BENCHMARK_CAPTURE_MODE=pipelined
      BENCHMARK_SENSOR_FRAME_RATE=65
      BENCHMARK_CLUSTER_OFFLOAD=0
      BENCHMARK_ENCODER=fc
      ;;
    qqvga-pipelined-65fps-gray4)
      BENCHMARK_RESOLUTION=qqvga
      BENCHMARK_CAPTURE_MODE=pipelined
      BENCHMARK_SENSOR_FRAME_RATE=65
      BENCHMARK_ENCODING=gray4
      BENCHMARK_CLUSTER_OFFLOAD=0
      BENCHMARK_ENCODER=none
      ;;
    qqvga-pipelined-120fps-gray4)
      BENCHMARK_RESOLUTION=qqvga
      BENCHMARK_CAPTURE_MODE=pipelined
      BENCHMARK_SENSOR_FRAME_RATE=120
      BENCHMARK_ENCODING=gray4
      BENCHMARK_CLUSTER_OFFLOAD=0
      BENCHMARK_ENCODER=none
      ;;
    qqvga-pipelined-100fps-gray4)
      BENCHMARK_RESOLUTION=qqvga
      BENCHMARK_CAPTURE_MODE=pipelined
      BENCHMARK_SENSOR_FRAME_RATE=100
      BENCHMARK_ENCODING=gray4
      BENCHMARK_CLUSTER_OFFLOAD=0
      BENCHMARK_ENCODER=none
      ;;
    qqvga-pipelined-120fps-gray4-48x36)
      BENCHMARK_RESOLUTION=qqvga
      BENCHMARK_CAPTURE_MODE=pipelined
      BENCHMARK_SENSOR_FRAME_RATE=120
      BENCHMARK_ENCODING=gray4
      BENCHMARK_STREAM_WIDTH=48
      BENCHMARK_STREAM_HEIGHT=36
      BENCHMARK_CLUSTER_OFFLOAD=0
      BENCHMARK_ENCODER=none
      ;;
    qqvga-pipelined-60fps-raw)
      BENCHMARK_RESOLUTION=qqvga
      BENCHMARK_CAPTURE_MODE=pipelined
      BENCHMARK_SENSOR_FRAME_RATE=60
      BENCHMARK_ENCODING=raw
      BENCHMARK_CLUSTER_OFFLOAD=0
      BENCHMARK_ENCODER=none
      ;;
    *)
      echo "Unknown benchmark variant: $1" >&2
      exit 2
      ;;
  esac
  if test "$BENCHMARK_ENCODING" = "gray4"; then
    BENCHMARK_IMAGE="$ARTIFACT_DIR/aideck-gap8-${BENCHMARK_RESOLUTION}-gray4-${BENCHMARK_STREAM_WIDTH}x${BENCHMARK_STREAM_HEIGHT}-${BENCHMARK_CAPTURE_MODE}-${BENCHMARK_SENSOR_FRAME_RATE}fps-maxfps.img"
  elif test "$BENCHMARK_ENCODING" = "raw"; then
    BENCHMARK_IMAGE="$ARTIFACT_DIR/aideck-gap8-${BENCHMARK_RESOLUTION}-raw-${BENCHMARK_CAPTURE_MODE}-${BENCHMARK_SENSOR_FRAME_RATE}fps-maxfps.img"
  elif test "$BENCHMARK_SENSOR_FRAME_RATE" != "default"; then
    BENCHMARK_IMAGE="$ARTIFACT_DIR/aideck-gap8-${BENCHMARK_RESOLUTION}-jpeg-${BENCHMARK_ENCODER}-${BENCHMARK_CAPTURE_MODE}-${BENCHMARK_SENSOR_FRAME_RATE}fps-maxfps.img"
  elif test "$BENCHMARK_CAPTURE_MODE" = "start-stop"; then
    BENCHMARK_IMAGE="$ARTIFACT_DIR/aideck-gap8-${BENCHMARK_RESOLUTION}-jpeg-${BENCHMARK_ENCODER}-maxfps.img"
  else
    BENCHMARK_IMAGE="$ARTIFACT_DIR/aideck-gap8-${BENCHMARK_RESOLUTION}-jpeg-${BENCHMARK_ENCODER}-${BENCHMARK_CAPTURE_MODE}-maxfps.img"
  fi
}

build_benchmark_variant() {
  local variant="$1"
  set_benchmark_variant "$variant"
  validate_sources
  docker run --rm \
    -v "$GAP8_DIR:/module" \
    bitcraze/aideck \
    tools/build/make-example \
    examples/other/wifi-img-streamer \
    clean build image \
    SETUP_WIFI_AP=1 \
    CAMERA_RESOLUTION="$BENCHMARK_RESOLUTION" \
    CAPTURE_MODE="$BENCHMARK_CAPTURE_MODE" \
    SENSOR_FRAME_RATE="$BENCHMARK_SENSOR_FRAME_RATE" \
    STREAM_ENCODING="$BENCHMARK_ENCODING" \
    STREAM_WIDTH="$BENCHMARK_STREAM_WIDTH" \
    STREAM_HEIGHT="$BENCHMARK_STREAM_HEIGHT" \
    JPEG_CLUSTER_OFFLOAD="$BENCHMARK_CLUSTER_OFFLOAD" \
    OUTPUT_PROFILING_DATA=1 \
    TARGET_FRAME_TIME_MS=0

  test -s "$GAP8_IMAGE"
  mkdir -p "$ARTIFACT_DIR"
  cp "$GAP8_IMAGE" "$BENCHMARK_IMAGE"
  shasum -a 256 "$BENCHMARK_IMAGE" > "$BENCHMARK_IMAGE.sha256"
  {
    echo "built_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "gap8_commit=$GAP8_COMMIT"
    echo "gap8_patch_sha256=$(shasum -a 256 "$PATCH" | awk '{print $1}')"
    echo "gap8_source_diff_sha256=$(shasum -a 256 "$PATCH" | awk '{print $1}')"
    echo "resolution=$BENCHMARK_RESOLUTION"
    echo "capture_mode=$BENCHMARK_CAPTURE_MODE"
    echo "sensor_frame_rate=$BENCHMARK_SENSOR_FRAME_RATE"
    echo "encoding=$BENCHMARK_ENCODING"
    echo "stream_width=$BENCHMARK_STREAM_WIDTH"
    echo "stream_height=$BENCHMARK_STREAM_HEIGHT"
    echo "jpeg_encoder=$BENCHMARK_ENCODER"
    echo "target_frame_time_ms=0"
  } > "$BENCHMARK_IMAGE.build-info"
  echo "Built GAP8 benchmark variant: $BENCHMARK_IMAGE"
}

build_benchmark_matrix() {
  local variant
  for variant in "${GAP8_BENCHMARK_VARIANTS[@]}"; do
    build_benchmark_variant "$variant"
  done
}

build_capture_matrix() {
  local variant
  for variant in "${GAP8_CAPTURE_VARIANTS[@]}"; do
    build_benchmark_variant "$variant"
  done
}

show_status() {
  for path in "$ESP_DIR" "$GAP8_DIR"; do
    if test -d "$path/.git"; then
      echo "$path: $(git -C "$path" rev-parse HEAD)"
    else
      echo "$path: missing"
    fi
  done
  for image in "$ESP_BOOTLOADER" "$ESP_IMAGE" "$GAP8_IMAGE"; do
    if test -s "$image"; then
      echo "$image: $(stat -f '%z bytes' "$image")"
    else
      echo "$image: missing"
    fi
  done
  if test -s "$GAP8_MAX_IMAGE"; then
    echo "$GAP8_MAX_IMAGE: $(stat -f '%z bytes' "$GAP8_MAX_IMAGE")"
  fi
  local variant
  for variant in "${GAP8_BENCHMARK_VARIANTS[@]}"; do
    set_benchmark_variant "$variant"
    if test -s "$BENCHMARK_IMAGE"; then
      echo "$BENCHMARK_IMAGE: $(stat -f '%z bytes' "$BENCHMARK_IMAGE")"
    fi
  done
  for variant in "${GAP8_CAPTURE_VARIANTS[@]}"; do
    set_benchmark_variant "$variant"
    if test -s "$BENCHMARK_IMAGE"; then
      echo "$BENCHMARK_IMAGE: $(stat -f '%z bytes' "$BENCHMARK_IMAGE")"
    fi
  done
}

flash_images() {
  test "${AIDECK_UDP_FLASH_CONFIRM:-}" = "FLASH_AIDECK_UDP" || {
    echo "Refusing to flash without AIDECK_UDP_FLASH_CONFIRM=FLASH_AIDECK_UDP" >&2
    exit 1
  }
  validate_sources
  test -s "$ESP_IMAGE" || {
    echo "Missing ESP32 image; run build first." >&2
    exit 1
  }
  test -s "$GAP8_STABLE_IMAGE" || {
    echo "Missing GAP8 image; run build first." >&2
    exit 1
  }
  flash_gap8_image "$GAP8_STABLE_IMAGE"
  "$CFLOADER" flash "$ESP_IMAGE" deck-bcAI:esp-fw -w "$AIDECK_URI"
  echo "UDP firmware flashed. Power-cycle the Crazyflie before testing."
}

flash_max_image() {
  flash_benchmark_variant qvga-fc
}

flash_semantic_image() {
  flash_prebuilt_image "$GAP8_SEMANTIC_SAFE_IMAGE" "semantic QVGA JPEG"
}

flash_semantic_highres_image() {
  flash_semantic_image
}

flash_policy_safe_image() {
  flash_prebuilt_image "$GAP8_POLICY_SAFE_IMAGE" "policy gray4"
}

flash_prebuilt_image() {
  local image="$1"
  local profile="$2"
  test "${AIDECK_UDP_FLASH_CONFIRM:-}" = "FLASH_AIDECK_UDP" || {
    echo "Refusing to flash without AIDECK_UDP_FLASH_CONFIRM=FLASH_AIDECK_UDP" >&2
    exit 1
  }
  test -s "$image" || {
    echo "Missing frame-safe image: $image" >&2
    exit 1
  }
  flash_gap8_image "$image"
  echo "Flashed and restarted frame-safe $profile profile."
}

flash_benchmark_variant() {
  local variant="$1"
  set_benchmark_variant "$variant"
  test "${AIDECK_UDP_FLASH_CONFIRM:-}" = "FLASH_AIDECK_UDP" || {
    echo "Refusing to flash without AIDECK_UDP_FLASH_CONFIRM=FLASH_AIDECK_UDP" >&2
    exit 1
  }
  validate_sources
  test -s "$BENCHMARK_IMAGE" || {
    echo "Missing benchmark image; run build-benchmark $variant first." >&2
    exit 1
  }
  flash_gap8_image "$BENCHMARK_IMAGE"
  echo "Flashed and restarted $variant. Verify links before testing."
}

case "${1:-}" in
  prepare)
    prepare_sources
    ;;
  preflight)
    validate_sources
    ;;
  build)
    build_images
    ;;
  build-max)
    build_max_image
    ;;
  build-benchmark)
    build_benchmark_variant "${2:-}"
    ;;
  build-matrix)
    build_benchmark_matrix
    ;;
  build-capture-matrix)
    build_capture_matrix
    ;;
  status)
    show_status
    ;;
  flash)
    flash_images
    ;;
  flash-max)
    flash_max_image
    ;;
  flash-benchmark)
    flash_benchmark_variant "${2:-}"
    ;;
  flash-semantic)
    flash_semantic_image
    ;;
  flash-semantic-highres)
    flash_semantic_highres_image
    ;;
  flash-policy-safe)
    flash_policy_safe_image
    ;;
  *)
    usage
    exit 2
    ;;
esac
