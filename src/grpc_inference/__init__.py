"""Standalone gRPC inference service package."""

from pathlib import Path
import sys

# Ensure generated protobuf modules can resolve the "inference.*" package.
proto_root = Path(__file__).parent / "proto"
if proto_root.exists():
    sys.path.append(str(proto_root))
