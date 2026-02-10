import os
import uuid

import grpc

from analytics.v1 import analytics_pb2, analytics_pb2_grpc


def _parse_timeout(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        timeout = float(value)
    except ValueError:
        return default
    if timeout <= 0:
        return default
    return timeout


class AnalyticsCRUDClient:
    def __init__(self, target: str | None = None, timeout_seconds: float | None = None):
        if target is None:
            target = os.getenv("ANALYTICS_CRUD_TARGET", "analytics:50051")
        if timeout_seconds is None:
            timeout_seconds = _parse_timeout(
                os.getenv("ANALYTICS_CRUD_TIMEOUT_SECONDS"),
                default=15.0,
            )
        self.target = target
        self.timeout_seconds = timeout_seconds
        self.channel = grpc.insecure_channel(self.target)
        self.stub = analytics_pb2_grpc.AnalyticsServiceStub(self.channel)

    def _get_metadata(self, request_id: str | None = None):
        if request_id is None:
            request_id = str(uuid.uuid4())
        return [("x-request-id", request_id)]

    def get_daily_stats(self, days: int = 30, request_id: str | None = None):
        request = analytics_pb2.GetDailyStatsRequest(days=days)
        return self.stub.GetDailyStats(
            request,
            timeout=self.timeout_seconds,
            metadata=self._get_metadata(request_id),
        )

    def get_transaction_details(
        self, days: int = 7, limit: int = 1000, request_id: str | None = None
    ):
        request = analytics_pb2.GetTransactionDetailsRequest(days=days, limit=limit)
        return self.stub.GetTransactionDetails(
            request,
            timeout=self.timeout_seconds,
            metadata=self._get_metadata(request_id),
        )

    def get_recent_alerts(self, limit: int = 50, request_id: str | None = None):
        request = analytics_pb2.GetRecentAlertsRequest(limit=limit)
        return self.stub.GetRecentAlerts(
            request,
            timeout=self.timeout_seconds,
            metadata=self._get_metadata(request_id),
        )

    def search_transactions(
        self,
        user_id: str = "",
        transaction_id: str = "",
        min_amount: float = None,
        max_amount: float = None,
        start_date: str = "",
        end_date: str = "",
        is_fraudulent: bool = None,
        limit: int = 100,
        offset: int = 0,
        request_id: str | None = None,
    ):
        request = analytics_pb2.SearchTransactionsRequest(
            user_id=user_id,
            transaction_id=transaction_id,
            min_amount=min_amount,
            max_amount=max_amount,
            start_date=start_date,
            end_date=end_date,
            is_fraudulent=is_fraudulent,
            limit=limit,
            offset=offset,
        )
        return self.stub.SearchTransactions(
            request,
            timeout=self.timeout_seconds,
            metadata=self._get_metadata(request_id),
        )

    def get_features(self, user_id: str, request_id: str | None = None):
        """Fetch latest features for a user via SearchTransactions."""
        response = self.search_transactions(
            user_id=user_id, limit=1, request_id=request_id
        )
        if response.transactions:
            return response.transactions[0]
        return None

    def get_training_data(self, cutoff_date, request_id: str | None = None):
        """Fetch training and test data via GetTrainingData."""
        request = analytics_pb2.GetTrainingDataRequest(cutoff_date=cutoff_date)
        return self.stub.GetTrainingData(
            request,
            timeout=self.timeout_seconds,
            metadata=self._get_metadata(request_id),
        )

    def store_generated_data(self, records, metadata, request_id: str | None = None):
        """Store generated records and metadata via StoreGeneratedData."""
        request = analytics_pb2.StoreGeneratedDataRequest(
            records=records, metadata=metadata
        )
        return self.stub.StoreGeneratedData(
            request,
            timeout=self.timeout_seconds,
            metadata=self._get_metadata(request_id),
        )

    def generate_data(
        self,
        num_users: int,
        fraud_rate: float,
        drop_existing: bool = False,
        seed: int = None,
        idempotency_key: str = "",
        request_id: str | None = None,
    ):
        """Generate synthetic data via GenerateData RPC (Go implementation)."""
        request = analytics_pb2.GenerateDataRequest(
            num_users=num_users,
            fraud_rate=fraud_rate,
            drop_existing=drop_existing,
            seed=seed,
            idempotency_key=idempotency_key,
        )
        try:
            return self.stub.GenerateData(
                request, timeout=300.0, metadata=self._get_metadata(request_id)
            )
        except grpc.RpcError as e:
            status_code = e.code()
            details = e.details()
            if status_code == grpc.StatusCode.INVALID_ARGUMENT:
                raise ValueError(f"Invalid generation parameters: {details}") from e
            if status_code == grpc.StatusCode.ALREADY_EXISTS:
                # Should be treated as conflict/in-progress for idempotency
                raise RuntimeError(f"Generation job conflict: {details}") from e
            if status_code == grpc.StatusCode.DEADLINE_EXCEEDED:
                raise TimeoutError(f"Generation timed out: {details}") from e
            if status_code == grpc.StatusCode.UNIMPLEMENTED:
                raise NotImplementedError(f"Generation not enabled: {details}") from e
            raise RuntimeError(f"Generation failed ({status_code}): {details}") from e

    def clear_all_data(self, request_id: str | None = None):
        """Clear all data via ClearAllData."""
        request = analytics_pb2.ClearAllDataRequest()
        return self.stub.ClearAllData(
            request,
            timeout=self.timeout_seconds,
            metadata=self._get_metadata(request_id),
        )

    def materialize_features(
        self, batch_size: int = 1000, request_id: str | None = None
    ):
        """Materialize features via MaterializeFeatures."""
        request = analytics_pb2.MaterializeFeaturesRequest(batch_size=batch_size)
        return self.stub.MaterializeFeatures(
            request,
            timeout=self.timeout_seconds,
            metadata=self._get_metadata(request_id),
        )

    def get_drift_window(self, hours: int = 24, request_id: str | None = None):
        """Fetch drift window data via GetDriftWindow."""
        request = analytics_pb2.GetDriftWindowRequest(hours=hours)
        return self.stub.GetDriftWindow(
            request,
            timeout=self.timeout_seconds,
            metadata=self._get_metadata(request_id),
        )

    def get_inference_scores(self, hours: int = 24, request_id: str | None = None):
        """Fetch inference scores via GetInferenceScores."""
        request = analytics_pb2.GetInferenceScoresRequest(hours=hours)
        return self.stub.GetInferenceScores(
            request,
            timeout=self.timeout_seconds,
            metadata=self._get_metadata(request_id),
        )

    def get_overview_metrics(self, request_id: str | None = None):
        request = analytics_pb2.GetOverviewMetricsRequest()
        return self.stub.GetOverviewMetrics(
            request,
            timeout=self.timeout_seconds,
            metadata=self._get_metadata(request_id),
        )

    def get_dataset_fingerprint(self, request_id: str | None = None):
        request = analytics_pb2.GetDatasetFingerprintRequest()
        return self.stub.GetDatasetFingerprint(
            request,
            timeout=self.timeout_seconds,
            metadata=self._get_metadata(request_id),
        )

    def get_feature_sample(
        self,
        sample_size: int = 100,
        stratify: bool = False,
        request_id: str | None = None,
    ):
        request = analytics_pb2.GetFeatureSampleRequest(
            sample_size=sample_size,
            stratify=stratify,
        )
        return self.stub.GetFeatureSample(
            request,
            timeout=self.timeout_seconds,
            metadata=self._get_metadata(request_id),
        )

    def get_schema_summary(
        self, table_names: list[str] = None, request_id: str | None = None
    ):
        if table_names is None:
            table_names = ["generated_records", "feature_snapshots"]
        request = analytics_pb2.GetSchemaSummaryRequest(table_names=table_names)
        return self.stub.GetSchemaSummary(
            request,
            timeout=self.timeout_seconds,
            metadata=self._get_metadata(request_id),
        )

    def report_training_run(self, run, request_id: str | None = None):
        """Report training run status to Analytics service."""
        request = analytics_pb2.ReportTrainingRunRequest(run=run)
        return self.stub.ReportTrainingRun(
            request,
            timeout=self.timeout_seconds,
            metadata=self._get_metadata(request_id),
        )


_client = None


def get_crud_client():
    global _client
    if _client is None:
        _client = AnalyticsCRUDClient()
    return _client


def reset_crud_client():
    """Reset the singleton client (for testing)."""
    global _client
    _client = None


def set_crud_client(client):
    """Set a custom client (for testing)."""
    global _client
    _client = client
