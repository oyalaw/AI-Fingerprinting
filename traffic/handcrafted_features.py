"""Handcrafted statistical feature extraction over a pcap's already-parsed
packet events (traffic/packet_reader.py's read_packets() output, shared
with every other exporter rather than each one re-parsing the pcap) --
the literature-standard fixed-length feature-vector representation for
traffic-fingerprinting classifiers (as opposed to traffic/sequence_export.py's
raw per-packet sequence, which a sequence-model classifier consumes
directly instead).

Computed once over the whole capture (row_type="overall") by default, or
per fixed-width time window (row_type="window", one row per window) when
window_seconds is set -- both written to the same CSV, distinguished by
row_type/window_index/window_start_sec/window_end_sec, so a downstream
reader can pick either granularity from one file.

72 features total, not 92: the original spec this module was built from
named 92 as the feature count but only actually listed 72 distinct
field names. Implemented here is exactly the 72 that were named -- see
this project's own conversation history for the count discrepancy this
was flagged against. Add more explicitly if a specific missing set is
identified, rather than padding the count with invented features.

TLS record detection is a documented, real approximation, not full TCP
stream reassembly: each TCP segment's payload is checked for a
plausible TLS record header (ContentType byte in {20,21,22,23}, version
major byte 0x03) at its own start. This correctly finds a record's
declared length without needing to decrypt anything, but a record whose
bytes are split across multiple packets is only counted once, at the
packet actually containing its header, and a record header that happens
to land mid-packet (after another complete record in the same segment)
is not searched for -- both real, known limitations of a header-only
detector versus full reassembly.
"""
import csv
import pathlib

import numpy as np
from scipy import stats as scipy_stats

_FULL_STATS = ("mean", "median", "std", "min", "max", "q25", "q75", "p90", "p95", "entropy", "skewness", "kurtosis")
_BASIC_STATS = ("mean", "std", "min", "max")


def _shannon_entropy(values, bins=20):
    """Histogram-based Shannon entropy (bits) -- values are continuous
    (packet sizes are already discrete integers so this naturally
    collapses to one bin per unique size; IATs are binned since they're
    real-valued)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.all(values == values[0]):
        return 0.0
    counts, _ = np.histogram(values, bins=min(bins, len(np.unique(values))))
    probs = counts[counts > 0] / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


def _describe(values, prefix, stats=_FULL_STATS):
    """Returns {prefix_stat: value} for each requested stat name, or
    {prefix_stat: None} for every stat if values is empty -- keeps the
    CSV's column set fixed regardless of how much data a given
    window/capture actually contains."""
    if not values:
        return {f"{prefix}_{s}": None for s in stats}
    arr = np.asarray(values, dtype=float)
    available = {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "entropy": _shannon_entropy(arr),
        # scipy's skew/kurtosis need >=2 distinct points; degenerate
        # (all-identical or single-value) inputs return 0.0 rather than
        # raising or emitting NaN.
        "skewness": float(scipy_stats.skew(arr)) if len(arr) > 1 else 0.0,
        "kurtosis": float(scipy_stats.kurtosis(arr)) if len(arr) > 1 else 0.0,
    }
    return {f"{prefix}_{s}": available[s] for s in stats}


def _extract_window(events, gap_threshold_s):
    """All 72 features for one slice of events (either the whole capture,
    or one fixed-width window of it)."""
    packet_count = len(events)
    sizes = [e["size"] for e in events]
    up_sizes = [e["size"] for e in events if e["direction"] == "up"]
    down_sizes = [e["size"] for e in events if e["direction"] == "down"]

    directions = [e["direction"] for e in events]
    direction_switches = sum(1 for a, b in zip(directions, directions[1:]) if a != b)
    duration = (events[-1]["ts"] - events[0]["ts"]) if len(events) > 1 else 0.0

    iats = [b["ts"] - a["ts"] for a, b in zip(events, events[1:])]
    iats = [max(iat, 0.0) for iat in iats]  # out-of-order capture timestamps shouldn't produce a negative IAT

    # Bursts: a new burst starts whenever the gap since the previous packet
    # exceeds gap_threshold_s -- same definition as traffic/burst_features.py,
    # replicated here (not imported) since this module also needs each
    # burst's dominant direction and the gaps *between* bursts (idle time),
    # neither of which that module computes.
    bursts = []
    idle_gaps = []
    current = None
    prev_ts = None
    for e in events:
        if current is None or (e["ts"] - prev_ts) > gap_threshold_s:
            if prev_ts is not None:
                idle_gaps.append(e["ts"] - prev_ts)
            current = {"start": e["ts"], "end": e["ts"], "packets": 0, "bytes": 0, "up": 0, "down": 0}
            bursts.append(current)
        current["end"] = e["ts"]
        current["packets"] += 1
        current["bytes"] += e["size"]
        current[e["direction"]] += 1
        prev_ts = e["ts"]

    burst_durations = [b["end"] - b["start"] for b in bursts]
    burst_intervals = [
        bursts[i + 1]["start"] - bursts[i]["end"] for i in range(len(bursts) - 1)
    ]
    burst_bytes = [b["bytes"] for b in bursts]
    burst_packets = [b["packets"] for b in bursts]
    burst_count_up = sum(1 for b in bursts if b["up"] >= b["down"])
    burst_count_down = sum(1 for b in bursts if b["down"] > b["up"])

    tcp_events = [e for e in events if e["proto"] == "tcp"]
    seen_retrans = set()
    retransmissions = 0
    for e in tcp_events:
        key = e["retrans_key"]
        if key is None:
            continue
        if key in seen_retrans:
            retransmissions += 1
        else:
            seen_retrans.add(key)

    tls_sizes = [e["tls_record_size"] for e in events if e["tls_record_size"] is not None]
    connections = {e["conn_key"] for e in tcp_events}

    row = {
        "packet_count_total": packet_count,
        "direction_switch_count": direction_switches,
        "direction_switch_rate": (direction_switches / packet_count) if packet_count else 0.0,
        "burst_count_total": len(bursts),
        "burst_count_up": burst_count_up,
        "burst_count_down": burst_count_down,
        "burst_frequency_per_second": (len(bursts) / duration) if duration > 0 else 0.0,
        "idle_gap_count": len(idle_gaps),
        "idle_time_total_sec": sum(idle_gaps),
        "idle_time_mean_sec": (sum(idle_gaps) / len(idle_gaps)) if idle_gaps else 0.0,
        "idle_time_max_sec": max(idle_gaps) if idle_gaps else 0.0,
        "tcp_packet_count": len(tcp_events),
        "udp_packet_count": sum(1 for e in events if e["proto"] == "udp"),
        "tcp_syn_count": sum(1 for e in tcp_events if e["flags"] & 0x02),
        "tcp_ack_count": sum(1 for e in tcp_events if e["flags"] & 0x10),
        "tcp_fin_count": sum(1 for e in tcp_events if e["flags"] & 0x01),
        "tcp_rst_count": sum(1 for e in tcp_events if e["flags"] & 0x04),
        "tcp_retransmission_count": retransmissions,
        "tls_record_count": len(tls_sizes),
        "connection_count": len(connections),
    }
    row.update(_describe(sizes, "packet_size", _FULL_STATS))
    row.update(_describe(up_sizes, "upload_packet_size", _BASIC_STATS))
    row.update(_describe(down_sizes, "download_packet_size", _BASIC_STATS))
    row.update(_describe(iats, "iat_sec", _FULL_STATS))
    row.update(_describe(burst_bytes, "burst_bytes", _BASIC_STATS))
    row.update(_describe(burst_packets, "burst_packets", _BASIC_STATS))
    row.update(_describe(burst_durations, "burst_duration_sec", _BASIC_STATS))
    row.update(_describe(burst_intervals, "burst_interval_sec", _BASIC_STATS))
    row.update(_describe(tls_sizes, "tls_record_size", _BASIC_STATS))
    return row


FIELDNAMES = (
    ["experiment_id", "row_type", "window_index", "window_start_sec", "window_end_sec"]
    + list(_extract_window([], gap_threshold_s=0.5).keys())
)


def extract_features(events, experiment_id, gap_threshold_s=0.5, window_seconds=None):
    """events is traffic/packet_reader.py's read_packets() output --
    already parsed once and shared across every exporter core/
    experiment.py calls, rather than each one re-parsing the same pcap.
    Returns a list of row dicts: one row_type="overall" row covering the
    whole capture, plus (if window_seconds is set) one row_type="window"
    row per fixed-width window, each computed independently over just
    that window's own packets."""
    rows = []
    overall = {"experiment_id": experiment_id, "row_type": "overall", "window_index": None,
               "window_start_sec": None, "window_end_sec": None}
    overall.update(_extract_window(events, gap_threshold_s))
    rows.append(overall)

    if window_seconds and events:
        t0 = events[0]["ts"]
        t_end = events[-1]["ts"]
        window_index = 0
        w_start = 0.0
        while w_start < (t_end - t0) or window_index == 0:
            w_end = w_start + window_seconds
            window_events = [e for e in events if w_start <= (e["ts"] - t0) < w_end]
            row = {
                "experiment_id": experiment_id,
                "row_type": "window",
                "window_index": window_index,
                "window_start_sec": w_start,
                "window_end_sec": w_end,
            }
            row.update(_extract_window(window_events, gap_threshold_s))
            rows.append(row)
            window_index += 1
            w_start = w_end

    return rows


def export_features(events, output_csv, experiment_id, gap_threshold_s=0.5, window_seconds=None):
    """Computes extract_features() and writes it to output_csv -- mirrors
    traffic/sequence_export.py's export_sequence()/traffic/flow_features.py's
    export_flow_features()/traffic/burst_features.py's export_bursts()
    shape, called from core/experiment.py alongside those."""
    rows = extract_features(events, experiment_id, gap_threshold_s=gap_threshold_s, window_seconds=window_seconds)

    output_csv = pathlib.Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return output_csv
