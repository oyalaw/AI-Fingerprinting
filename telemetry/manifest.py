"""Feature-extraction provenance manifest: <experiment_id>_manifest.json.

Deliberately excludes this project's own ground-truth labels (framework,
family, architecture, application, dataset, device -- all already in
ground_truth.json) -- not because this file represents "the attacker's
observation point" (it never gets sniffed off the wire, so that framing
doesn't actually hold), but because it's provenance for the *feature-
extraction pipeline itself*: which extractor version ran, what parameters
it used, a hash of the exact pcap bytes it read. Keeping labels out of it
is what makes it safe to hand a classifier pipeline this file plus the
feature CSVs without also handing it the answer key by accident.
"""
import datetime
import hashlib
import json
import pathlib
import platform
import socket

from traffic.handcrafted_features import FIELDNAMES as HANDCRAFTED_FIELDNAMES
from traffic.packet_reader import STANDARD_ETHERNET_FRAME_BYTES
from traffic.sequence_export import FIELDNAMES as SEQUENCE_FIELDNAMES


def _sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _best_effort_local_ip():
    # No general, always-correct way to learn "this process's own outbound
    # IP" -- gethostbyname(gethostname()) is a well-known unreliable trick
    # on multi-homed machines, but it's a reasonable best-effort fallback
    # rather than leaving this blank whenever it happens to work.
    try:
        return socket.gethostbyname(socket.gethostname())
    except socket.error:
        return None


def build_manifest(
    experiment_id,
    role,
    server_host,
    pcap_path=None,
    capture=None,
    sequence_csv=None,
    features_csv=None,
    feature_row_count=None,
    burst_gap_sec=0.5,
    window_seconds=None,
    oversized_threshold_bytes=STANDARD_ETHERNET_FRAME_BYTES,
):
    """capture is the ScapyCapture instance itself (or None if capture was
    disabled) -- reads its public capture_start_epoch/capture_end_epoch/
    written_packet_count attributes rather than duplicating that
    bookkeeping here."""
    pcap_path = pathlib.Path(pcap_path) if pcap_path else None
    pcap_exists = bool(pcap_path and pcap_path.exists())

    return {
        "experiment_id": experiment_id,
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "schemas": {
            "packet_sequence": list(SEQUENCE_FIELDNAMES),
            "handcrafted_features": list(HANDCRAFTED_FIELDNAMES),
        },
        "capture": {
            "pcap_path": str(pcap_path) if pcap_path else None,
            "pcap_size_bytes": pcap_path.stat().st_size if pcap_exists else None,
            "pcap_sha256": _sha256_of(pcap_path) if pcap_exists else None,
            "packet_count": capture.written_packet_count if capture else None,
            "capture_start_epoch": capture.capture_start_epoch if capture else None,
            "capture_end_epoch": capture.capture_end_epoch if capture else None,
        },
        "direction_reference": {
            "server_ip": server_host,
            # Only meaningful from the client side -- a server bound to
            # 0.0.0.0/a specific host doesn't know which client will
            # connect ahead of time, so this stays null on role=server.
            "client_ip": _best_effort_local_ip() if role == "client" else None,
        },
        "feature_parameters": {
            "burst_gap_sec": burst_gap_sec,
            # traffic/handcrafted_features.py uses the same gap threshold
            # for both burst boundaries and idle-gap detection -- not two
            # independently configurable values today, so this
            # deliberately mirrors burst_gap_sec rather than implying a
            # separate knob exists.
            "idle_threshold_sec": burst_gap_sec,
            "window_seconds": window_seconds,
            # Above this size (bytes), traffic/packet_reader.py flags a
            # captured "packet" as is_oversized -- a likely TSO/GRO capture
            # artifact rather than a real on-wire frame. Recorded here so a
            # downstream reader knows exactly what threshold produced the
            # oversized_packet_* columns in handcrafted_features.
            "oversized_packet_threshold_bytes": oversized_threshold_bytes,
        },
        "outputs": {
            "packet_sequence_csv": str(sequence_csv) if sequence_csv else None,
            "features_csv": str(features_csv) if features_csv else None,
            "feature_row_count": feature_row_count,
        },
        "extractor": {
            "parser": "scapy",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
    }


def write_manifest(path, manifest):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
