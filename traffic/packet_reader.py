"""Single shared pcap parse, reused by every downstream feature exporter.

Real, measured problem this fixes: traffic/sequence_export.py,
traffic/flow_features.py, traffic/burst_features.py, and
traffic/handcrafted_features.py each independently called scapy's
rdpcap() on the *same* pcap -- fine for the small captures most of this
project's own testing produced, but a genuine cost on a large one (one
real capture during this project's own testing: 717MB, 577,123 packets)
-- rdpcap() loads the entire pcap into memory as Python objects, known to
use far more RAM than the raw file size, and doing that four separate
times against the same large file is real, avoidable overhead that
plausibly contributes to a slow or failed run on a big enough capture.
Parsed once here instead; core/experiment.py passes the resulting event
list to all four exporters.
"""
try:
    from scapy.all import IP, TCP, UDP, rdpcap

    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False


def _require_scapy():
    if not SCAPY_AVAILABLE:
        raise RuntimeError("scapy is not installed; cannot read pcap files.")


def read_packets(pcap_path, server_port):
    """One dict per TCP/UDP packet (every other IP payload type, and any
    non-IP packet, is dropped -- consistent with what every consumer of
    this data already only ever looked at): ts, direction, size, proto,
    TCP flags, seq, a connection key (for connection counting), a
    retransmission key (only set for segments that actually carry a
    payload -- see traffic/handcrafted_features.py's own docstring for
    why bare ACKs must not count), and tls_record_size when this packet's
    payload starts with a plausible TLS record header."""
    _require_scapy()
    packets = rdpcap(str(pcap_path))
    events = []
    for pkt in packets:
        if IP not in pkt:
            continue
        ip = pkt[IP]
        if TCP in pkt:
            tcp = pkt[TCP]
            proto = "tcp"
            direction = "up" if tcp.dport == server_port else "down"
            sport, dport = tcp.sport, tcp.dport
            flags = int(tcp.flags)
            seq = int(tcp.seq)
            payload = bytes(tcp.payload) if tcp.payload else b""
        elif UDP in pkt:
            udp = pkt[UDP]
            proto = "udp"
            direction = "up" if udp.dport == server_port else "down"
            sport, dport = udp.sport, udp.dport
            flags = 0
            seq = None
            payload = b""
        else:
            continue

        tls_record_size = None
        if len(payload) >= 5 and payload[0] in (20, 21, 22, 23) and payload[1] == 3:
            declared_len = (payload[3] << 8) | payload[4]
            if 0 < declared_len <= 16709:
                tls_record_size = declared_len + 5

        events.append(
            {
                "ts": float(pkt.time),
                "direction": direction,
                "size": len(pkt),
                "proto": proto,
                "flags": flags,
                "seq": seq,
                "conn_key": frozenset({(ip.src, sport), (ip.dst, dport)}),
                "retrans_key": (ip.src, sport, ip.dst, dport, seq) if (seq is not None and payload) else None,
                "tls_record_size": tls_record_size,
            }
        )
    events.sort(key=lambda e: e["ts"])
    return events
