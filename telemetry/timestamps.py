import time


class RequestTimer:
    """Records monotonic per-request timing for a client or server loop."""

    def __init__(self):
        self._records = []
        self._open = {}

    def start(self, request_id):
        self._open[request_id] = time.perf_counter()

    def stop(self, request_id, **extra):
        started = self._open.pop(request_id, None)
        ended = time.perf_counter()
        record = {
            "request_id": request_id,
            "wall_time": time.time(),
            "duration_s": (ended - started) if started is not None else None,
            **extra,
        }
        self._records.append(record)
        return record

    @property
    def records(self):
        return list(self._records)
