#!/usr/bin/env sh
set -eu

PORT="${PORT:-8000}"
MODEL_PATH="${DEEPFAKE_MODEL_PATH:-${DEEPFAKE_WEIGHTS_DIR:-/app/backend/weights}/dfdc_effnet_b7.pt}"

if [ -n "${DEEPFAKE_MODEL_URL:-}" ] && [ ! -f "$MODEL_PATH" ]; then
  mkdir -p "$(dirname "$MODEL_PATH")"
  echo "Downloading model checkpoint to $MODEL_PATH"
  curl -L --fail --retry 3 --output "$MODEL_PATH" "$DEEPFAKE_MODEL_URL"
fi

exec python -m uvicorn backend.app:app --host 0.0.0.0 --port "$PORT"
