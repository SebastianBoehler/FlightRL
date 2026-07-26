#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-}"
DURATION_S="${AIDECK_GATE_DURATION_S:-300}"
AIDECK_HOST="${AIDECK_HOST:-192.168.4.1}"
TARGET_FPS="${AIDECK_GATE_FPS:-3}"
FRAME_COUNT=$((DURATION_S * TARGET_FPS))
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$ROOT/artifacts/ai_deck/udp_ground_gate_${MODE}_${STAMP}"

usage() {
  cat <<'EOF'
Usage: scripts/aideck_udp_ground_gate.sh stream-only|radio

The default gate is 300 seconds at 3 FPS. Override with:
  AIDECK_GATE_DURATION_S=60
  AIDECK_GATE_FPS=8
  AIDECK_HOST=192.168.4.1
EOF
}

capture_vision() {
  python "$ROOT/scripts/capture_aideck_vision.py" \
    --transport udp \
    --host "$AIDECK_HOST" \
    --frames "$FRAME_COUNT" \
    --timeout-s 5 \
    --width 64 \
    --height 48 \
    --include-delta \
    --include-motion-mask \
    --frame-dir "$RUN_DIR/frames" \
    --output "$RUN_DIR/vision.npz"
}

mkdir -p "$RUN_DIR"
case "$MODE" in
  stream-only)
    capture_vision
    ;;
  radio)
    python "$ROOT/scripts/crazyflie_log.py" \
      --config "$ROOT/configs/hardware/crazyflie_2_1_brushless_flow_only.toml" \
      --duration-s "$DURATION_S" \
      --output "$RUN_DIR/telemetry.csv" \
      --console-output "$RUN_DIR/console.jsonl" &
    telemetry_pid=$!
    trap 'kill "$telemetry_pid" 2>/dev/null || true' EXIT
    sleep 2
    capture_vision
    wait "$telemetry_pid"
    trap - EXIT
    ;;
  *)
    usage
    exit 2
    ;;
esac

echo "AI Deck UDP ground gate completed: $RUN_DIR"
