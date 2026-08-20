"""Per-round, per-FL-phase correlation of classification metrics with
network-traffic and power/resource cost -- <experiment_id>_round_features.csv.

Deliberately a DIFFERENT kind of artifact than traffic/handcrafted_features.py's
own *_features.csv: that file is the attacker-facing traffic-fingerprinting
feature set, which must stay realistic to what a passive network observer
could actually see -- see telemetry/resource_monitor.py's own docstring on
why host-side power/GPU signals are deliberately kept out of it. This file
is the opposite, explanatory/analysis artifact (same category as
manifest.json/ground_truth.json): understanding what a FL run's own
per-round fit/evaluate/upload/download phases actually cost in bytes,
watts, and classification quality. Never treat this file as classifier
input alongside features.csv.

Built entirely post-hoc (matching traffic/packet_reader.py's own "parse
once, at the end" philosophy) from three artifacts every FL client
experiment already produces by the time it finishes: the client-role
event log (fl_frameworks/flower_adapter.py and fl_frameworks/
fedlab_adapter.py both stamp real epoch timestamps on every fit/evaluate/
download/upload phase boundary, plus real accuracy/precision_score/
recall/f1_score at fl_evaluate_end -- core/classification_metrics.py),
the parsed packet events (traffic/packet_reader.py's read_packets()), and
the resource telemetry CSV rows (telemetry/resource_monitor.py). No new
live/in-process hooks into either subsystem.

Phase-window shape is deliberately adaptive per FL framework rather than
one fixed template, because the two currently-instrumented frameworks
have genuinely different real RPC shapes, confirmed directly by reading
each adapter's own event sequence:

  Flower: download(fit) -> fit -> upload_ready -> download(evaluate) ->
  evaluate  -- two separate server RPCs per round (fit, then evaluate on
  the newly-aggregated model), each preceded by its own real download.

  FedLab: download(fit) -> fit -> evaluate -> upload_ready  -- FedLab's
  basic FedAvg pattern has no separate server-driven evaluate RPC at all
  (confirmed directly reading fedlab.contrib.algorithm.basic_server/
  basic_client -- only ParameterUpdate messages ever cross the wire),
  so fl_frameworks/fedlab_adapter.py evaluates immediately after fit,
  in-process, and uploads once at the very end of the round.

"download" and "upload" windows are approximate round-trip boundaries,
not exact wire-transfer durations -- neither adapter's client-side code
can observe the other side's send/receive boundary directly, only its
own before/after (see each phase's own comment below). "fit" and
"evaluate" are exact: both boundaries are this same process's own
timestamps around a loop it runs itself.
"""
import csv
import datetime
import pathlib

_RESOURCE_NUMERIC_FIELDS = (
    "cpu_usage_percent",
    "memory_usage_mb",
    "memory_usage_percent",
    "gpu_usage_percent",
    "gpu_memory_used_mb",
    "cpu_power_w",
    "gpu_power_w",
    "system_power_w",
)

FIELDNAMES = (
    ["experiment_id", "round", "phase", "sub_phase", "window_start_ts", "window_end_ts", "window_duration_sec"]
    + ["loss", "examples", "accuracy", "precision_score", "recall", "f1_score"]
    + ["traffic_packet_count", "traffic_byte_count", "traffic_upload_bytes", "traffic_download_bytes",
       "traffic_oversized_packet_count"]
    + [f"resource_{field}_mean" for field in _RESOURCE_NUMERIC_FIELDS]
)


def load_phase_events(events_path):
    """Reads a client_events.jsonl back in -- one dict per line, already
    including the real epoch `ts` telemetry/experiment_log.py stamps on
    every record."""
    import json

    events_path = pathlib.Path(events_path)
    if not events_path.exists():
        return []
    records = []
    with events_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _find(records, event_type, round_number=None, phase=None):
    for r in records:
        if r.get("event") != event_type:
            continue
        if round_number is not None and r.get("round") != round_number:
            continue
        if phase is not None and r.get("phase") != phase:
            continue
        return r
    return None


def build_round_phase_windows(records):
    """Returns a list of {round, phase, sub_phase, start_ts, end_ts,
    loss, examples, accuracy, precision_score, recall, f1_score} dicts --
    only the keys that phase actually has are non-None. See this module's
    own docstring for why the exact set of windows produced per round
    depends on which markers that round's FL framework actually logged."""
    rounds = sorted({r["round"] for r in records if r.get("round") is not None})
    client_start = _find(records, "fl_client_start")
    windows = []
    prev_round_end_ts = client_start["ts"] if client_start else None

    for round_number in rounds:
        download_fit = _find(records, "fl_download_complete", round_number, phase="fit")
        fit_start = _find(records, "fl_fit_start", round_number)
        fit_end = _find(records, "fl_fit_end", round_number)
        upload_ready = _find(records, "fl_upload_ready", round_number)
        download_eval = _find(records, "fl_download_complete", round_number, phase="evaluate")
        eval_start = _find(records, "fl_evaluate_start", round_number)
        eval_end = _find(records, "fl_evaluate_end", round_number)

        # download(fit): real download of this round's global model, plus
        # whatever wait preceded it (server-side sampling/aggregation from
        # the previous round, or process startup for round 1) -- bundled
        # together since client-side code can't separate them.
        if download_fit and prev_round_end_ts is not None:
            windows.append(
                {"round": round_number, "phase": "download", "sub_phase": "fit",
                 "start_ts": prev_round_end_ts, "end_ts": download_fit["ts"]}
            )

        if fit_start and fit_end:
            windows.append(
                {"round": round_number, "phase": "fit", "sub_phase": None,
                 "start_ts": fit_start["ts"], "end_ts": fit_end["ts"],
                 "loss": fit_end.get("loss"), "examples": fit_end.get("examples")}
            )

        if download_eval:
            # Flower shape: a real separate evaluate RPC exists, so the
            # observable "upload" window is just the local return-overhead
            # between finishing fit and this process handing results back
            # to Flower's own ClientApp -- the real wire send (and the
            # server's aggregation) happens after upload_ready and is
            # folded into this round's second download window instead,
            # since it isn't separably observable from here.
            if fit_end and upload_ready:
                windows.append(
                    {"round": round_number, "phase": "upload", "sub_phase": None,
                     "start_ts": fit_end["ts"], "end_ts": upload_ready["ts"]}
                )
            if upload_ready:
                windows.append(
                    {"round": round_number, "phase": "download", "sub_phase": "evaluate",
                     "start_ts": upload_ready["ts"], "end_ts": download_eval["ts"]}
                )
        # FedLab shape: no separate evaluate RPC/download at all -- evaluate
        # runs immediately after fit in the same local call, and the one
        # real upload for the whole round happens at the very end, after
        # evaluate completes.
        elif eval_end and upload_ready:
            windows.append(
                {"round": round_number, "phase": "upload", "sub_phase": None,
                 "start_ts": eval_end["ts"], "end_ts": upload_ready["ts"]}
            )

        if eval_start and eval_end:
            windows.append(
                {"round": round_number, "phase": "evaluate", "sub_phase": None,
                 "start_ts": eval_start["ts"], "end_ts": eval_end["ts"],
                 "loss": eval_end.get("loss"), "accuracy": eval_end.get("accuracy"),
                 "precision_score": eval_end.get("precision_score"),
                 "recall": eval_end.get("recall"), "f1_score": eval_end.get("f1_score")}
            )

        prev_round_end_ts = (eval_end or upload_ready or fit_end or download_fit or {}).get("ts", prev_round_end_ts)

    return windows


def _traffic_summary(packet_events, start_ts, end_ts):
    """Compact traffic stats for a short phase window -- deliberately not
    the full 72-stat traffic/handcrafted_features.py treatment (skewness/
    percentiles are unstable or meaningless over the handful of packets a
    single FL phase typically produces)."""
    window_events = [e for e in packet_events if start_ts <= e["ts"] < end_ts]
    return {
        "traffic_packet_count": len(window_events),
        "traffic_byte_count": sum(e["size"] for e in window_events),
        "traffic_upload_bytes": sum(e["size"] for e in window_events if e["direction"] == "up"),
        "traffic_download_bytes": sum(e["size"] for e in window_events if e["direction"] == "down"),
        "traffic_oversized_packet_count": sum(1 for e in window_events if e.get("is_oversized")),
    }


def _parse_resource_timestamp(row):
    # resource_monitor.py's own relative_time_sec uses time.monotonic(),
    # not comparable to the event log's time.time()-based `ts` -- only
    # timestamp_utc (a real wall-clock ISO string) can be aligned with it.
    try:
        return datetime.datetime.fromisoformat(row["timestamp_utc"]).timestamp()
    except (KeyError, ValueError):
        return None


def _resource_summary(resource_rows_with_epoch, start_ts, end_ts):
    """Mean of each numeric resource_monitor.py column over the rows whose
    (derived) epoch timestamp falls in a phase window. A column with no
    non-blank values in-window (e.g. gpu_power_w on a machine with no
    NVML) stays None -- see telemetry/resource_monitor.py's own docstring
    for why some columns are legitimately blank per-device, not a bug."""
    window_rows = [row for epoch, row in resource_rows_with_epoch if epoch is not None and start_ts <= epoch < end_ts]
    summary = {}
    for field in _RESOURCE_NUMERIC_FIELDS:
        values = []
        for row in window_rows:
            raw = row.get(field)
            if raw not in (None, ""):
                try:
                    values.append(float(raw))
                except ValueError:
                    continue
        summary[f"resource_{field}_mean"] = (sum(values) / len(values)) if values else None
    return summary


def build_round_phase_features(experiment_id, phase_events, packet_events, resource_rows):
    """phase_events: load_phase_events()'s own output (this experiment's
    own client_events.jsonl). packet_events: traffic/packet_reader.py's
    read_packets() output. resource_rows: list of dict rows read back
    from telemetry/resource_monitor.py's own _resource.csv (e.g. via
    csv.DictReader). Returns one row per (round, phase[, sub_phase])."""
    windows = build_round_phase_windows(phase_events)
    resource_rows_with_epoch = [(_parse_resource_timestamp(row), row) for row in resource_rows]

    rows = []
    for window in windows:
        row = {
            "experiment_id": experiment_id,
            "round": window["round"],
            "phase": window["phase"],
            "sub_phase": window.get("sub_phase"),
            "window_start_ts": window["start_ts"],
            "window_end_ts": window["end_ts"],
            "window_duration_sec": window["end_ts"] - window["start_ts"],
            "loss": window.get("loss"),
            "examples": window.get("examples"),
            "accuracy": window.get("accuracy"),
            "precision_score": window.get("precision_score"),
            "recall": window.get("recall"),
            "f1_score": window.get("f1_score"),
        }
        row.update(_traffic_summary(packet_events, window["start_ts"], window["end_ts"]))
        row.update(_resource_summary(resource_rows_with_epoch, window["start_ts"], window["end_ts"]))
        rows.append(row)
    return rows


def export_round_phase_features(experiment_id, phase_events, packet_events, resource_rows, output_csv):
    rows = build_round_phase_features(experiment_id, phase_events, packet_events, resource_rows)
    output_csv = pathlib.Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return output_csv, rows
