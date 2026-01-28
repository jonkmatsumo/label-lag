import grpc
import pytest

from grpc_inference.config import GRPCInferenceConfig
from grpc_inference.service import InferenceService
from grpc_inference.proto.inference.v1 import inference_pb2


class FakeContext(grpc.ServicerContext):
    def __init__(self):
        self._code = None
        self._details = None

    def abort(self, code, details):
        self._code = code
        self._details = details
        raise grpc.RpcError(details)

    def abort_with_status(self, status):
        self._code = status.code
        self._details = status.details
        raise grpc.RpcError(status.details)

    def set_code(self, code):
        self._code = code

    def set_details(self, details):
        self._details = details

    def invocation_metadata(self):
        return []

    def peer(self):
        return ""

    def peer_identities(self):
        return None

    def peer_identity_key(self):
        return None

    def auth_context(self):
        return {}

    def send_initial_metadata(self, initial_metadata):
        return None

    def set_trailing_metadata(self, trailing_metadata):
        return None

    def is_active(self):
        return True

    def time_remaining(self):
        return 1.0

    def cancel(self):
        return False

    def add_callback(self, callback):
        return False

    def disable_next_message_compression(self):
        return None


@pytest.fixture()
def service():
    config = GRPCInferenceConfig(
        host="0.0.0.0",
        port=50052,
        db_dsn=None,
        mlflow_tracking_uri=None,
        model_name=None,
        model_stage="Production",
        include_features_used=False,
        score_timeout_ms=1500,
        config_path=None,
    )
    return InferenceService(config)


def test_score_requires_user_id(service):
    ctx = FakeContext()
    request = inference_pb2.ScoreRequest(
        user_id="", amount=10.0, currency="USD", client_transaction_id="txn"
    )
    with pytest.raises(grpc.RpcError):
        service.Score(request, ctx)


def test_score_requires_client_transaction_id(service):
    ctx = FakeContext()
    request = inference_pb2.ScoreRequest(
        user_id="user", amount=10.0, currency="USD", client_transaction_id=""
    )
    with pytest.raises(grpc.RpcError):
        service.Score(request, ctx)
