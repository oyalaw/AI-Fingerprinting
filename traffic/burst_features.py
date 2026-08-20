"""Burst segmentation: group packets into bursts separated by inter-arrival
gaps above a threshold — a common feature-engineering step in traffic-
fingerprinting research."""
import csv
import pathlib


def extract_bursts(events, gap_threshold_s=0.5):
    """events is traffic/packet_reader.py's read_packets() output --
    already parsed once and shared across every exporter core/
    experiment.py calls, rather than each one re-parsing the same pcap.
    events are already sorted by timestamp by the reader."""
    bursts = []
    current = None
    prev_ts = None
    for e in events:
        ts, size = e["ts"], e["size"]
        if current is None or (ts - prev_ts) > gap_threshold_s:
            current = {"start": ts, "end": ts, "packet_count": 0, "byte_count": 0}
            bursts.append(current)
        current["end"] = ts
        current["packet_count"] += 1
        current["byte_count"] += size
        prev_ts = ts

    for b in bursts:
        b["duration_s"] = b["end"] - b["start"]
    return bursts


def export_bursts(events, output_csv, gap_threshold_s=0.5):
    """Computes extract_bursts() and writes it to output_csv -- mirrors
    traffic/sequence_export.py's export_sequence()/
    traffic/flow_features.py's export_flow_features() shape, called from
    core/experiment.py alongside those."""
    bursts = extract_bursts(events, gap_threshold_s=gap_threshold_s)

    output_csv = pathlib.Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["start", "end", "packet_count", "byte_count", "duration_s"])
        for b in bursts:
            writer.writerow([b["start"], b["end"], b["packet_count"], b["byte_count"], b["duration_s"]])
    return output_csv
