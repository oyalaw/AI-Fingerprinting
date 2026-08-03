"""Burst segmentation: group packets into bursts separated by inter-arrival
gaps above a threshold — a common feature-engineering step in traffic-
fingerprinting research."""
try:
    from scapy.all import IP, TCP, rdpcap

    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False


def _require_scapy():
    if not SCAPY_AVAILABLE:
        raise RuntimeError("scapy is not installed; cannot read pcap files.")


def extract_bursts(pcap_path, gap_threshold_s=0.5):
    _require_scapy()
    packets = rdpcap(str(pcap_path))
    events = sorted(
        (float(pkt.time), len(pkt)) for pkt in packets if IP in pkt and TCP in pkt
    )

    bursts = []
    current = None
    prev_ts = None
    for ts, size in events:
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
