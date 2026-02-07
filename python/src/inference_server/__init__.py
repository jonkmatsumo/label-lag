"""Standalone gRPC inference service package."""

import sys
from pathlib import Path

# Ensure generated protobuf modules can resolve the "inference.*" package.
proto_root = Path(__file__).parent / "proto"
if proto_root.exists():
    sys.path.append(str(proto_root))
