import json
import os
import unittest
from unittest.mock import MagicMock, patch

import grpc

from analytics.v1 import analytics_pb2
from training.crud_client import AnalyticsCRUDClient


class TestAnalyticsCRUDClientRetry(unittest.TestCase):
    def setUp(self):
        self.client = AnalyticsCRUDClient(target="localhost:50051")
        self.mock_run = analytics_pb2.TrainingRun(
            run_id="test-run",
            model_name="test-model",
            status="FINISHED",
        )

    @patch("training.crud_client.AnalyticsCRUDClient._get_metadata")
    def test_report_training_run_retries_and_succeeds(self, mock_metadata):
        mock_metadata.return_value = [("x-request-id", "test-req")]

        # Mock stub to fail twice then succeed
        self.client.stub.ReportTrainingRun = MagicMock()
        self.client.stub.ReportTrainingRun.side_effect = [
            grpc.RpcError(),  # First call fails
            grpc.RpcError(),  # Second call fails
            analytics_pb2.ReportTrainingRunResponse(success=True),  # Third succeeds
        ]

        # We need to ensure the RpcError has a retryable code
        error = grpc.RpcError()
        error.code = lambda: grpc.StatusCode.UNAVAILABLE
        self.client.stub.ReportTrainingRun.side_effect = [
            error,
            error,
            analytics_pb2.ReportTrainingRunResponse(success=True),
        ]

        # Patch tenacity wait to avoid delay in tests
        with patch("tenacity.nap.time.sleep", return_value=None):
            resp = self.client.report_training_run(self.mock_run)

        self.assertIsNotNone(resp)
        self.assertTrue(resp.success)
        self.assertEqual(self.client.stub.ReportTrainingRun.call_count, 3)

    @patch("training.crud_client.os.makedirs")
    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    @patch("training.crud_client.AnalyticsCRUDClient._get_metadata")
    def test_report_training_run_fallback_on_permanent_failure(
        self, mock_metadata, mock_open, mock_makedirs
    ):
        mock_metadata.return_value = [("x-request-id", "test-req")]

        # Mock stub to always fail with a retryable error
        class MockRpcError(grpc.RpcError):
            def code(self):
                return grpc.StatusCode.UNAVAILABLE

            def details(self):
                return "unavailable"

        self.client.stub.ReportTrainingRun = MagicMock(side_effect=MockRpcError())

        # Patch tenacity wait and stop to speed up test
        with patch("tenacity.nap.time.sleep", return_value=None):
            with patch(
                "training.crud_client.stop_after_delay", return_value=lambda x: True
            ):
                resp = self.client.report_training_run(self.mock_run)

        self.assertIsNone(resp)
        # Check if fallback was called
        mock_open.assert_called()
        # Verify JSON content
        handle = mock_open()
        args, kwargs = handle.write.call_args
        written_data = json.loads(args[0].strip())
        self.assertEqual(written_data["run_id"], "test-run")

    def test_fallback_persist_writes_to_correct_path(self):
        with patch("training.crud_client.os.makedirs") as mock_makedirs:
            with patch("builtins.open", unittest.mock.mock_open()) as mock_file:
                self.client._fallback_persist(self.mock_run, "req-123")

                mock_makedirs.assert_called()
                # Check that open was called with a path ending in
                # var/training_run_reports.jsonl
                args, kwargs = mock_file.call_args
                path = args[0]
                self.assertTrue(
                    path.endswith(os.path.join("var", "training_run_reports.jsonl"))
                )
                self.assertEqual(args[1], "a")

    @patch("training.crud_client.os.path.exists")
    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    @patch("training.crud_client.os.remove")
    def test_replay_spooled_reports_success(self, mock_remove, mock_open, mock_exists):
        mock_exists.return_value = True
        payload = {
            "run_id": "spooled-run",
            "model_name": "test-model",
            "status": "COMPLETED",
            "metrics": "{}",
            "params": "{}",
        }
        mock_open.return_value.readlines.return_value = [json.dumps(payload) + "\n"]

        self.client.stub.ReportTrainingRun = MagicMock(
            return_value=analytics_pb2.ReportTrainingRunResponse(success=True)
        )

        self.client.replay_spooled_reports()

        self.assertEqual(self.client.stub.ReportTrainingRun.call_count, 1)
        mock_remove.assert_called_once()
