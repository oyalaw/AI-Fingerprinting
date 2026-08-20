import datetime
import json
import pathlib
import time


class ExperimentLog:
    """Structured JSONL event log for a single experiment run.

    common_fields (experiment_id, role, framework, architecture, etc. --
    see core/experiment.py's _common_event_fields()) are stamped onto
    every single record, not just inference ones, so each line in this
    file is independently readable without needing a join against
    ground_truth.json -- a deliberate denormalization tradeoff (repeating
    ~14 fields on every line costs real file size for a long, high-
    frequency run) chosen so this file can be handed to a downstream
    reader on its own."""

    def __init__(self, path, common_fields=None):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.common_fields = common_fields or {}

    def event(self, event_type, **fields):
        now = time.time()
        record = {
            **self.common_fields,
            "event": event_type,
            "timestamp_utc": datetime.datetime.fromtimestamp(now, tz=datetime.timezone.utc).isoformat(),
            "ts": now,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return record
