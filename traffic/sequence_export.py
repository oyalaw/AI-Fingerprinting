"""Export a pcap as an ordered (timestamp, direction, size) sequence CSV —
the standard packet-size/direction sequence representation used across the
traffic-fingerprinting literature, and the primary feature format this
project produces for downstream classifiers."""
import csv
import pathlib

try:
    from scapy.all import IP, TCP, rdpcap

    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

FIELDNAMES = ["timestamp", "direction", "size_bytes"]


def export_sequence(pcap_path, output_csv, server_port):
    if not SCAPY_AVAILABLE:
        raise RuntimeError("scapy is not installed; cannot read pcap files.")

    packets = rdpcap(str(pcap_path))
    rows = []
    for pkt in packets:
        if IP not in pkt or TCP not in pkt:
            continue
        direction = "up" if pkt[TCP].dport == server_port else "down"
        rows.append((float(pkt.time), direction, len(pkt)))
    rows.sort(key=lambda r: r[0])

    output_csv = pathlib.Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDNAMES)
        writer.writerows(rows)
    return output_csv
