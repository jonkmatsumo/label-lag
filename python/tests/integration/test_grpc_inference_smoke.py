import os

import grpc
import pytest

from grpc_inference.proto.inference.v1 import inference_pb2, inference_pb2_grpc

GRPC_INFERENCE_ADDR = os.getenv("GRPC_INFERENCE_ADDR", "localhost:50052")


@pytest.fixture(scope="module")
def grpc_channel():
    channel = grpc.insecure_channel(GRPC_INFERENCE_ADDR)
    try:
        grpc.channel_ready_future(channel).result(timeout=5)
    except Exception as exc:
        pytest.skip(f"gRPC inference service not reachable: {exc}")
    return channel


def test_score_smoke(grpc_channel):
    stub = inference_pb2_grpc.InferenceServiceStub(grpc_channel)
    request = inference_pb2.ScoreRequest(
        user_id="user_001",
        amount=100.0,
        currency="USD",
        client_transaction_id="txn_smoke",
    )
    response = stub.Score(request, timeout=10)
    assert response.model_version
    assert response.request_id
