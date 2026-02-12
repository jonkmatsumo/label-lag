from __future__ import annotations

from datetime import UTC, datetime

import grpc
import pytest

from training.job_queue import JobQueue
from training.job_store import InMemoryJobStore
from training.jobs import TrialRecord, TuningJob
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


def _build_service_with_trials() -> tuple[TrainingService, str]:
    store = InMemoryJobStore()
    queue = JobQueue()
    service = TrainingService(store, queue)

    job = TuningJob.create(config={"feature_columns": ["f1"]}, total_trials=5)
    store.create(job)

    now = datetime.now(UTC)
    values = [0.4, 0.7, 0.2, 0.9, 0.5]
    for idx, value in enumerate(values):
        store.append_trial(
            job.job_id,
            TrialRecord(
                trial_number=idx,
                state="TrialState.COMPLETE",
                value=value,
                params={"max_depth": str(idx)},
                started_at=now,
                ended_at=now,
                duration_ms=100.0 + idx,
            ),
        )

    return service, job.job_id


def test_list_trials_supports_cursor_pagination():
    service, job_id = _build_service_with_trials()

    first_page = service.ListTrials(
        training_pb2.ListTrialsRequest(
            job_id=job_id,
            sort_by="trial_number",
            limit=2,
        ),
        FakeContext(),
    )
    second_page = service.ListTrials(
        training_pb2.ListTrialsRequest(
            job_id=job_id,
            sort_by="trial_number",
            limit=2,
            cursor=first_page.next_cursor,
        ),
        FakeContext(),
    )

    assert [trial.trial_number for trial in first_page.trials] == [0, 1]
    assert first_page.next_cursor == "1"
    assert [trial.trial_number for trial in second_page.trials] == [2, 3]
    assert second_page.next_cursor == "3"


def test_list_trials_supports_top_k_sorting_by_value():
    service, job_id = _build_service_with_trials()

    response = service.ListTrials(
        training_pb2.ListTrialsRequest(
            job_id=job_id,
            sort_by="value",
            limit=3,
        ),
        FakeContext(),
    )

    assert [trial.trial_number for trial in response.trials] == [3, 1, 4]
    assert response.next_cursor == ""


def test_list_trials_rejects_cursor_for_value_sort():
    service, job_id = _build_service_with_trials()

    with pytest.raises(FakeRpcAbortError) as exc:
        service.ListTrials(
            training_pb2.ListTrialsRequest(
                job_id=job_id,
                sort_by="value",
                limit=2,
                cursor="1",
            ),
            FakeContext(),
        )

    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
