#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_OUT_DIR="$ROOT_DIR/src/api/proto"
GO_OUT_DIR="$ROOT_DIR/src/services/analytics-crud/proto"

mkdir -p "$PYTHON_OUT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

echo "Generating Python stubs..."
"$PYTHON_BIN" -m grpc_tools.protoc \
  -I"$ROOT_DIR/src/services/analytics-crud/proto" \
  --python_out="$PYTHON_OUT_DIR" \
  --grpc_python_out="$PYTHON_OUT_DIR" \
  crud/v1/analytics.proto

echo "Generating Go stubs..."
if command -v protoc >/dev/null && command -v protoc-gen-go >/dev/null && command -v protoc-gen-go-grpc >/dev/null; then
  protoc -I "$ROOT_DIR/src/services/analytics-crud/proto" \
    --go_out="$GO_OUT_DIR" --go_opt=paths=source_relative \
    --go-grpc_out="$GO_OUT_DIR" --go-grpc_opt=paths=source_relative \
    crud/v1/analytics.proto
else
  echo "Warning: protoc or go plugins not found, skipping Go stub generation"
fi

# Ensure packages are importable
find "$PYTHON_OUT_DIR" -type d -exec touch {}/__init__.py \;