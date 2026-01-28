import os

import grpc

from api.proto.proto.crud.v1 import analytics_pb2, analytics_pb2_grpc


class AnalyticsCRUDClient:
    def __init__(self, target: str = None):
        if target is None:
            target = os.getenv("ANALYTICS_CRUD_TARGET", "analytics-crud:50051")
        self.target = target
        self.channel = grpc.insecure_channel(self.target)
        self.stub = analytics_pb2_grpc.AnalyticsServiceStub(self.channel)

    def get_daily_stats(self, days: int = 30):
        request = analytics_pb2.GetDailyStatsRequest(days=days)
        return self.stub.GetDailyStats(request)

    def get_transaction_details(self, days: int = 7, limit: int = 1000):
        request = analytics_pb2.GetTransactionDetailsRequest(days=days, limit=limit)
        return self.stub.GetTransactionDetails(request)

    def get_recent_alerts(self, limit: int = 50):
        request = analytics_pb2.GetRecentAlertsRequest(limit=limit)
        return self.stub.GetRecentAlerts(request)

    def get_overview_metrics(self):
        request = analytics_pb2.GetOverviewMetricsRequest()
        return self.stub.GetOverviewMetrics(request)

    def get_dataset_fingerprint(self):
        request = analytics_pb2.GetDatasetFingerprintRequest()
        return self.stub.GetDatasetFingerprint(request)

    def get_feature_sample(self, sample_size: int = 100, stratify: bool = False):
        request = analytics_pb2.GetFeatureSampleRequest(
            sample_size=sample_size,
            stratify=stratify,
        )
        return self.stub.GetFeatureSample(request)

    def get_schema_summary(self, table_names: list[str] = None):
        if table_names is None:
            table_names = ["generated_records", "feature_snapshots"]
        request = analytics_pb2.GetSchemaSummaryRequest(table_names=table_names)
        return self.stub.GetSchemaSummary(request)


_client = None


def get_crud_client():
    global _client
    if _client is None:
        _client = AnalyticsCRUDClient()
    return _client
