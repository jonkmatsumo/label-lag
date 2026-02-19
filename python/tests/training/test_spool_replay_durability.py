"""Regression tests for spool replay durability and TOCTOU safety."""

import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from analytics.v1 import analytics_pb2
from training.crud_client import AnalyticsCRUDClient


class TestSpoolReplayDurability:
    @pytest.fixture
    def var_dir(self, tmp_path):
        """Create a temporary var directory."""
        vdir = tmp_path / "var"
        vdir.mkdir()
        # Mock os.getcwd to return tmp_path so the client uses our temp var dir
        with patch("os.getcwd", return_value=str(tmp_path)):
            yield vdir

    def test_replay_atomic_rename_safety(self, var_dir):
        """Verify that concurrent appends during replay are not lost."""
        log_path = var_dir / "training_run_reports.jsonl"

        # 1. Create initial spool with one report
        report1 = {
            "run_id": "run-1",
            "model_name": "model-1",
            "status": "COMPLETED",
            "metrics": "{}",
            "params": "{}",
            "tenant_id": "tenant-1",
            "request_id": "req-1",
            "timestamp": time.time(),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(report1) + "\n")

        client = AnalyticsCRUDClient(target="localhost:50051")
        client.stub = MagicMock()

        # 2. Mock ReportTrainingRun to simulate a concurrent append while replay
        # is happening. We'll use a side_effect to append to the file.
        def concurrent_append(*args, **kwargs):
            report2 = {
                "run_id": "run-2",
                "model_name": "model-1",
                "status": "COMPLETED",
                "metrics": "{}",
                "params": "{}",
                "tenant_id": "tenant-1",
                "request_id": "req-2",
                "timestamp": time.time(),
            }
            # New entries go to the *live* spool path
            with open(log_path, "a") as f:
                f.write(json.dumps(report2) + "\n")
            return analytics_pb2.ReportTrainingRunResponse()

        client.stub.ReportTrainingRun.side_effect = concurrent_append

        # 3. Run replay
        client.replay_spooled_reports()

        # 4. Assertions
        # - Report 1 should have been replayed
        assert client.stub.ReportTrainingRun.call_count == 1

        # - Live spool file should still exist and contain ONLY Report 2
        # (The processing file should be gone, and any failures would be appended back)
        assert os.path.exists(log_path)
        with open(log_path) as f:
            lines = f.readlines()

        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["run_id"] == "run-2"

        # - Processing file should be deleted
        assert not os.path.exists(str(log_path) + ".processing")

    def test_failed_replay_lines_are_appended_back(self, var_dir):
        """Verify that lines failing during replay are restored to the live spool."""
        log_path = var_dir / "training_run_reports.jsonl"

        report1 = {
            "run_id": "fail-1",
            "model_name": "m",
            "status": "S",
            "metrics": "{}",
            "params": "{}",
            "tenant_id": "t",
            "request_id": "r",
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(report1) + "\n")

        client = AnalyticsCRUDClient(target="localhost:50051")
        client.stub = MagicMock()
        # Fail the RPC
        client.stub.ReportTrainingRun.side_effect = Exception("RPC failed")

        client.replay_spooled_reports()

        # Both the original report and any new ones should be in the spool
        assert os.path.exists(log_path)
        with open(log_path) as f:
            lines = f.readlines()

        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["run_id"] == "fail-1"
