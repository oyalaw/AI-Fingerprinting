"""pcap -> flow-level features: byte/packet counts and duration per capture,
split by direction relative to the experiment's server port."""
import json
import pathlib


def extract_flow_features(events):
    """events is traffic/packet_reader.py's read_packets() output --
    already parsed once and shared across every exporter core/
    experiment.py calls, rather than each one re-parsing the same pcap."""
    up_bytes = up_packets = down_bytes = down_packets = 0
    first_ts = last_ts = None

    for e in events:
        ts = e["ts"]
        first_ts = ts if first_ts is None else min(first_ts, ts)
        last_ts = ts if last_ts is None else max(last_ts, ts)
        if e["direction"] == "up":
            up_bytes += e["size"]
            up_packets += 1
        else:
            down_bytes += e["size"]
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


def export_flow_features(events, output_json):
    """Computes extract_flow_features() and writes it to output_json --
    mirrors traffic/sequence_export.py's export_sequence()/
    traffic/burst_features.py's export_bursts() shape, called from
    core/experiment.py alongside those."""
    features = extract_flow_features(events)

    output_json = pathlib.Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(features, indent=2), encoding="utf-8")
    return output_json
