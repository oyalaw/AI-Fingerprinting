import json
import pathlib
import time


class ExperimentLog:
    """Structured JSONL event log for a single experiment run."""

    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, event_type, **fields):
        record = {"ts": time.time(), "event": event_type, **fields}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return record
