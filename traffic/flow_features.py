"""pcap -> flow-level features: byte/packet counts and duration per capture,
split by direction relative to the experiment's server port."""
import json
import pathlib

try:
    from scapy.all import IP, TCP, rdpcap

    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False


def _require_scapy():
    if not SCAPY_AVAILABLE:
        raise RuntimeError("scapy is not installed; cannot read pcap files.")


def extract_flow_features(pcap_path, server_port):
    _require_scapy()
    packets = rdpcap(str(pcap_path))
    up_bytes = up_packets = down_bytes = down_packets = 0
    first_ts = last_ts = None

    for pkt in packets:
        if IP not in pkt or TCP not in pkt:
            continue
        ts = float(pkt.time)
        first_ts = ts if first_ts is None else min(first_ts, ts)
        last_ts = ts if last_ts is None else max(last_ts, ts)
        size = len(pkt)
        if pkt[TCP].dport == server_port:
            up_bytes += size
            up_packets += 1
        else:
            down_bytes += size
            down_packets += 1

    duration = (last_ts - first_ts) if first_ts is not None else 0.0
    return {
        "packet_count": up_packets + down_packets,
        "byte_count": up_bytes + down_bytes,
        "up_packets": up_packets,
        "up_bytes": up_bytes,
        "down_packets": down_packets,
        "down_bytes": down_bytes,
        "duration_s": duration,
    }


def export_flow_features(pcap_path, output_json, server_port):
    """Computes extract_flow_features() and writes it to output_json --
    mirrors traffic/sequence_export.py's export_sequence()/
    traffic/burst_features.py's export_bursts() shape, called from
    core/experiment.py alongside those."""
    features = extract_flow_features(pcap_path, server_port)

    output_json = pathlib.Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(features, indent=2), encoding="utf-8")
    return output_json
