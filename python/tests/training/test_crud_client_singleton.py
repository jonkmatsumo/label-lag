import threading
from concurrent.futures import ThreadPoolExecutor

from training import crud_client


def test_get_crud_client_is_thread_safe_singleton(monkeypatch):
    crud_client.reset_crud_client()
    constructed = []

    class FakeClient:
        def __init__(self):
            constructed.append(object())

    monkeypatch.setattr(crud_client, "AnalyticsCRUDClient", FakeClient)

    num_threads = 20
    barrier = threading.Barrier(num_threads)

    def _worker():
        barrier.wait()
        return crud_client.get_crud_client()

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(_worker) for _ in range(num_threads)]
        instances = [future.result() for future in futures]

    assert len(constructed) == 1
    first = instances[0]
    assert all(instance is first for instance in instances)

    crud_client.reset_crud_client()
