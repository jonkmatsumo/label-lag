import queue
import threading


class JobQueue:
    def __init__(self):
        self._queue: queue.Queue[str] = queue.Queue()
        self._canceled: set[str] = set()
        self._lock = threading.Lock()

    def enqueue(self, job_id: str) -> None:
        self._queue.put(job_id)

    def cancel(self, job_id: str) -> None:
        with self._lock:
            self._canceled.add(job_id)

    def get(self, block: bool = True, timeout: float | None = None) -> str | None:
        # Safety limit to prevent infinite loop
        for _ in range(1000):
            try:
                job_id = self._queue.get(block=block, timeout=timeout)
            except queue.Empty:
                return None
            with self._lock:
                if job_id in self._canceled:
                    self._canceled.remove(job_id)
                    self._queue.task_done()
                    continue
            return job_id
        return None

    def task_done(self) -> None:
        self._queue.task_done()

    def depth(self) -> int:
        with self._lock:
            return max(0, self._queue.qsize() - len(self._canceled))
