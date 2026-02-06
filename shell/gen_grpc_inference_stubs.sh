#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTO_SRC="$ROOT_DIR/src/services/inference-gateway/proto/inference/v1/inference.proto"
OUT_DIR="$ROOT_DIR/src/grpc_inference/proto"

mkdir -p "$OUT_DIR/inference/v1"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

"$PYTHON_BIN" -m grpc_tools.protoc \
  -I"$ROOT_DIR/src/services/inference-gateway/proto" \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  "$PROTO_SRC"

# Ensure packages are importable
for pkg in "$OUT_DIR" "$OUT_DIR/inference" "$OUT_DIR/inference/v1"; do
  if [ ! -f "$pkg/__init__.py" ]; then
    echo '"""Generated protobuf package."""' > "$pkg/__init__.py"
  fi
done
