import queue


class JobQueue:
    def __init__(self):
        self._queue: queue.Queue[str] = queue.Queue()

    def enqueue(self, job_id: str) -> None:
        self._queue.put(job_id)

    def get(self, block: bool = True, timeout: float | None = None) -> str | None:
        try:
            return self._queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def task_done(self) -> None:
        self._queue.task_done()

    def depth(self) -> int:
        return self._queue.qsize()
