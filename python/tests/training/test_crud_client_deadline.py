"""Tests for retry/deadline interaction guardrails (Phase C)."""

import time
import unittest
from unittest.mock import MagicMock, patch

import grpc

from training.crud_client import (
    _RETRYABLE_READ_CODES,
    AnalyticsCRUDClient,
    _is_retryable_read,
)


class _MockRpcError(grpc.RpcError):
    def __init__(self, code):
        self._code = code

    def code(self):
        return self._code

    def details(self):
        return str(self._code)


class TestRetryableReadCodes(unittest.TestCase):
    """Ensure only UNAVAILABLE and DEADLINE_EXCEEDED are retried."""

    def test_unavailable_is_retryable(self):
        err = _MockRpcError(grpc.StatusCode.UNAVAILABLE)
        self.assertTrue(_is_retryable_read(err))

    def test_deadline_exceeded_is_retryable(self):
        err = _MockRpcError(grpc.StatusCode.DEADLINE_EXCEEDED)
        self.assertTrue(_is_retryable_read(err))

    def test_not_found_is_not_retryable(self):
        err = _MockRpcError(grpc.StatusCode.NOT_FOUND)
        self.assertFalse(_is_retryable_read(err))

    def test_invalid_argument_is_not_retryable(self):
        err = _MockRpcError(grpc.StatusCode.INVALID_ARGUMENT)
        self.assertFalse(_is_retryable_read(err))

    def test_internal_is_not_retryable(self):
        err = _MockRpcError(grpc.StatusCode.INTERNAL)
        self.assertFalse(_is_retryable_read(err))

    def test_permission_denied_is_not_retryable(self):
        err = _MockRpcError(grpc.StatusCode.PERMISSION_DENIED)
        self.assertFalse(_is_retryable_read(err))

    def test_non_rpc_error_is_not_retryable(self):
        err = RuntimeError("boom")
        self.assertFalse(_is_retryable_read(err))

    def test_retryable_codes_set_is_only_two(self):
        self.assertEqual(
            _RETRYABLE_READ_CODES,
            {grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED},
        )


class TestEffectiveTimeout(unittest.TestCase):
    """Test _effective_timeout respects caller deadline."""

    def setUp(self):
        self.client = AnalyticsCRUDClient(target="localhost:50051")

    def test_no_caller_deadline_uses_default(self):
        self.assertAlmostEqual(
            self.client._effective_timeout(),
            self.client.timeout_seconds,
            places=1,
        )

    def test_caller_deadline_far_future(self):
        # 30s from now — should use the configured default (15s)
        self.client.set_caller_deadline(time.time() + 30)
        timeout = self.client._effective_timeout()
        self.assertAlmostEqual(timeout, self.client.timeout_seconds, places=1)

    def test_caller_deadline_short_budget(self):
        # 1s from now — should clamp to 1s (< default 15s)
        self.client.set_caller_deadline(time.time() + 1.0)
        timeout = self.client._effective_timeout()
        self.assertLess(timeout, 2.0)
        self.assertGreater(timeout, 0.5)

    def test_caller_deadline_near_expiry(self):
        # 0.1s from now — less than min budget, should return min
        self.client.set_caller_deadline(time.time() + 0.1)
        timeout = self.client._effective_timeout()
        self.assertAlmostEqual(timeout, 0.3, places=1)

    def test_clear_caller_deadline(self):
        self.client.set_caller_deadline(time.time() + 1.0)
        self.client.clear_caller_deadline()
        self.assertAlmostEqual(
            self.client._effective_timeout(),
            self.client.timeout_seconds,
            places=1,
        )


class TestRetryWithNearDeadline(unittest.TestCase):
    """When deadline is nearly expired, retries should be minimal."""

    def setUp(self):
        self.client = AnalyticsCRUDClient(target="localhost:50051")

    @patch("tenacity.nap.time.sleep", return_value=None)
    def test_retryable_read_does_not_retry_on_non_retryable(self, _mock_sleep):
        """Non-retryable code should raise immediately with no retries."""
        error = _MockRpcError(grpc.StatusCode.NOT_FOUND)
        self.client.stub.GetDailyStats = MagicMock(side_effect=error)

        with self.assertRaises(grpc.RpcError):
            self.client.get_daily_stats()

        # Should be called exactly once (no retries for NOT_FOUND)
        self.assertEqual(self.client.stub.GetDailyStats.call_count, 1)


class TestGetFeaturesNotRetried(unittest.TestCase):
    """Audit: get_features must NOT have the retryable_read decorator."""

    def test_get_features_has_no_retry_decorator(self):
        # get_features delegates to search_transactions which IS retried,
        # but get_features itself should not have a double retry layer.
        client = AnalyticsCRUDClient(target="localhost:50051")
        method = client.get_features
        # Check it's not wrapped by tenacity
        self.assertFalse(
            hasattr(method, "retry"),
            "get_features should NOT have a tenacity retry decorator",
        )


class TestWriteMethodsNotRetried(unittest.TestCase):
    """Audit: write methods must NOT have the retryable_read decorator."""

    def test_generate_data_has_no_retry(self):
        client = AnalyticsCRUDClient(target="localhost:50051")
        self.assertFalse(hasattr(client.generate_data, "retry"))

    def test_clear_all_data_has_no_retry(self):
        client = AnalyticsCRUDClient(target="localhost:50051")
        self.assertFalse(hasattr(client.clear_all_data, "retry"))

    def test_materialize_features_has_no_retry(self):
        client = AnalyticsCRUDClient(target="localhost:50051")
        self.assertFalse(hasattr(client.materialize_features, "retry"))

    def test_store_generated_data_has_no_retry(self):
        client = AnalyticsCRUDClient(target="localhost:50051")
        self.assertFalse(hasattr(client.store_generated_data, "retry"))
