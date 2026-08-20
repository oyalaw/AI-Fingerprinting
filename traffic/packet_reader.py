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


# Standard non-jumbo Ethernet frame: 14-byte header + 1500-byte MTU payload.
# Every healthy capture in this project's own testing topped out exactly
# at this size -- a real capture on a machine with TSO/GRO NIC offload
# active showed a "packet" of 63,778 bytes instead, confirmed as a real,
# known Linux capture artifact, not a bug in this project's own code: TSO
# (TCP Segmentation Offload) hands the NIC one large send buffer and lets
# the *hardware* split it into wire-sized frames, and GRO (Generic Receive
# Offload) does the reverse on receive -- both happen at a point in the
# stack a raw AF_PACKET capture sits on the wrong side of, so it sees one
# oversized "packet" that was never actually one frame on the wire. A
# genuine passive network observer (this project's whole threat model)
# would never see that -- only the true, MTU-sized frames. Flagged here
# rather than silently filtered or re-split: reconstructing the real
# per-frame boundaries would require assuming the actual negotiated MSS,
# itself a guess, so a downstream reader gets to decide how to handle it
# rather than have this project quietly reshape the data on an assumption.
STANDARD_ETHERNET_FRAME_BYTES = 1514


def read_packets(pcap_path, server_port, oversized_threshold=STANDARD_ETHERNET_FRAME_BYTES):
    """One dict per TCP/UDP packet (every other IP payload type, and any
    non-IP packet, is dropped -- consistent with what every consumer of
    this data already only ever looked at): ts, direction, size, proto,
    TCP flags, seq, a connection key (for connection counting), a
    retransmission key (only set for segments that actually carry a
    payload -- see traffic/handcrafted_features.py's own docstring for
    why bare ACKs must not count), tls_record_size when this packet's
    payload starts with a plausible TLS record header, and is_oversized
    when size exceeds oversized_threshold (see this module's own
    STANDARD_ETHERNET_FRAME_BYTES docstring -- a likely TSO/GRO capture
    artifact, not a real on-wire frame)."""
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
                "is_oversized": len(pkt) > oversized_threshold,
            }
        )
    events.sort(key=lambda e: e["ts"])
    return events
