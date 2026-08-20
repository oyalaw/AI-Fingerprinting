"""Export a pcap as an ordered (timestamp, direction, size) sequence CSV —
the standard packet-size/direction sequence representation used across the
traffic-fingerprinting literature, and the primary feature format this
project produces for downstream classifiers."""
import csv
import pathlib

FIELDNAMES = ["timestamp", "direction", "size_bytes"]


def export_sequence(events, output_csv):
    """events is traffic/packet_reader.py's read_packets() output --
    already parsed once and shared across every exporter core/
    experiment.py calls, rather than each one re-parsing the same pcap."""
    rows = [(e["ts"], e["direction"], e["size"]) for e in events]

    output_csv = pathlib.Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDNAMES)
        writer.writerows(rows)
    return output_csv
