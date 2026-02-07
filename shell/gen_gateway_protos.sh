#!/usr/bin/env bash
set -euo pipefail

# Directories
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTO_ROOT="$ROOT_DIR/go/inference/proto"
PYTHON_OUT="$ROOT_DIR/python/src/gateway_grpc"
GO_OUT="$ROOT_DIR/go/inference/internal/http"  # Match option go_package

# Ensure output directories exist
mkdir -p "$PYTHON_OUT"
mkdir -p "$GO_OUT"

# Add Go bin to PATH (fallback)
GOBIN=$(go env GOPATH)/bin
export PATH="$PATH:$GOBIN"
PLUGIN_GO="$GOBIN/protoc-gen-go"
PLUGIN_GRPC="$GOBIN/protoc-gen-go-grpc"

# Python generation (using uv run to ensure dependencies)
echo "Generating Python stubs..."
uv run python -m grpc_tools.protoc \
  -I"$ROOT_DIR" \
  -I"$PROTO_ROOT" \
  --python_out="$PYTHON_OUT" \
  --grpc_python_out="$PYTHON_OUT" \
  "$PROTO_ROOT/gateway/v1/gateway.proto"

# Go generation
echo "Generating Go stubs..."

# Setup local symlinks for robustness
ln -sf "$GOBIN/protoc-gen-go" ./protoc-gen-go
ln -sf "$GOBIN/protoc-gen-go-grpc" ./protoc-gen-go-grpc
# Create underscore aliases just in case
ln -sf "$GOBIN/protoc-gen-go" ./protoc-gen-go_grpc
ln -sf "$GOBIN/protoc-gen-go-grpc" ./protoc-gen-go_grpc

export PATH=".:$PATH"

# Go generation
echo "Generating Go stubs..."
# Rely on PATH for plugins
# Use paths=source_relative to generate in the same directory as the proto
# Then we can move them if needed, or better yet, make the proto location match the package match
# But since proto is in proto/gateway/v1 and we want code in internal/http/gatewayv1,
# we might need to exact path mapping.
# Simplest: Generate to ROOT_DIR module-relative, which creates github.com/... structure,
# then we just move it or leave it if we change go_package.

# Let's try module mapping method if possible, or just generate and move.
# Moving is robust.
protoc \
  -I"$ROOT_DIR" \
  -I"$PROTO_ROOT" \
  --go_out="$ROOT_DIR" \
  --go_grpc_out="$ROOT_DIR" \
  "$PROTO_ROOT/gateway/v1/gateway.proto"

# Move generated files from nested structure to target
TARGET_DIR="$ROOT_DIR/go/inference/internal/http/gatewayv1"
mkdir -p "$TARGET_DIR"
# The generated path based on go_package "github.com/jonkmatsumo/label-lag/go/inference/internal/http/gatewayv1"
GEN_PATH="$ROOT_DIR/github.com/jonkmatsumo/label-lag/go/inference/internal/http/gatewayv1"

if [ -d "$GEN_PATH" ]; then
    echo "Moving generated Go files to $TARGET_DIR..."
    mv "$GEN_PATH"/*.go "$TARGET_DIR/"
    rm -rf "$ROOT_DIR/github.com"
else
    echo "Warning: Expected generated files at $GEN_PATH not found."
fi

# Make Python package importable
touch "$PYTHON_OUT/__init__.py"
echo "Done."
