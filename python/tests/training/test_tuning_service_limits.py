from __future__ import annotations

import grpc
import pytest

from training.job_queue import JobQueue
from training.job_store import InMemoryJobStore
from training.jobs import TuningJob
from training.service import TrainingService
from training.v1 import training_pb2


class FakeRpcAbortError(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode, details: str):
        super().__init__()
        self._code = code
        self._details = details

    def code(self):
        return self._code

    def details(self):
        return self._details


class FakeContext:
    def abort(self, code: grpc.StatusCode, details: str):
        raise FakeRpcAbortError(code=code, details=details)


def _request(
    n_trials: int = 5,
    timeout_minutes: int = 30,
    split_strategy: str = "temporal",
) -> training_pb2.TrainRequest:
    return training_pb2.TrainRequest(
        training_window_days=30,
        selected_feature_columns=["f1"],
        feature_resolution_mode="strict",
        split_config=training_pb2.SplitConfig(
            strategy=split_strategy,
            validation_fraction=0.2,
            seed=42,
        ),
        tuning_config=training_pb2.TuningConfig(
            enabled=True,
            strategy="bayesian",
            n_trials=n_trials,
            timeout_minutes=timeout_minutes,
            metric="pr_auc",
            direction="maximize",
        ),
    )


def test_start_tuning_rejects_trials_over_server_limit(monkeypatch):
    monkeypatch.setattr("training.service.DataLoader.FEATURE_COLUMNS", ["f1"])
    monkeypatch.setenv("MAX_TUNING_TRIALS", "2")

    service = TrainingService(InMemoryJobStore(), JobQueue())
    context = FakeContext()

    with pytest.raises(FakeRpcAbortError) as exc:
        service.StartTuningJob(_request(n_trials=3), context)

    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "n_trials exceeds server limit" in exc.value.details()


def test_start_tuning_rejects_timeout_over_server_limit(monkeypatch):
    monkeypatch.setattr("training.service.DataLoader.FEATURE_COLUMNS", ["f1"])
    monkeypatch.setenv("MAX_TUNING_TIMEOUT_MINUTES", "10")

    service = TrainingService(InMemoryJobStore(), JobQueue())
    context = FakeContext()

    with pytest.raises(FakeRpcAbortError) as exc:
        service.StartTuningJob(_request(timeout_minutes=11), context)

    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "timeout_minutes exceeds server limit" in exc.value.details()


def test_start_tuning_rejects_when_concurrency_cap_reached(monkeypatch):
    monkeypatch.setattr("training.service.DataLoader.FEATURE_COLUMNS", ["f1"])
    monkeypatch.setenv("MAX_CONCURRENT_TUNING_JOBS", "1")

    store = InMemoryJobStore()
    store.create(TuningJob.create(config={"test": 1}, total_trials=1))

    service = TrainingService(store, JobQueue())
    context = FakeContext()

    with pytest.raises(FakeRpcAbortError) as exc:
        service.StartTuningJob(_request(), context)

    assert exc.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
    assert "Maximum concurrent tuning jobs reached" in exc.value.details()


def test_validate_rejects_unsupported_split_strategy_when_strict(monkeypatch):
    monkeypatch.setattr("training.service.DataLoader.FEATURE_COLUMNS", ["f1"])
    monkeypatch.setenv("STRICT_SPLIT_STRATEGY_VALIDATION", "1")

    service = TrainingService(InMemoryJobStore(), JobQueue())
    context = FakeContext()

    with pytest.raises(FakeRpcAbortError) as exc:
        service.ValidateTrainRequest(
            _request(split_strategy="temporal_stratified"),
            context,
        )

    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "Unsupported split strategy 'temporal_stratified'" in exc.value.details()
    assert "Supported strategies" in exc.value.details()


def test_validate_warn_only_mode_allows_unsupported_split_strategy(monkeypatch):
    monkeypatch.setattr("training.service.DataLoader.FEATURE_COLUMNS", ["f1"])
    monkeypatch.setenv("STRICT_SPLIT_STRATEGY_VALIDATION", "0")

    service = TrainingService(InMemoryJobStore(), JobQueue())
    context = FakeContext()

    response = service.ValidateTrainRequest(
        _request(split_strategy="expanding_window"),
        context,
    )

    assert response.valid is True
    assert any("compatibility mode" in warning for warning in response.warnings)
