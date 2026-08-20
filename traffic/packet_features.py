"""scapy-based background packet capture, scoped to one experiment's
host:port. This is the primary data source for the whole project: everything
downstream (flow/burst features, sequence export, handcrafted features) reads
the .pcap this writes.

Requires a working packet-capture driver:
  - Windows: Npcap (https://npcap.com/), running as Administrator.
  - Linux/macOS: libpcap (usually preinstalled) + root or CAP_NET_RAW
    (`sudo setcap cap_net_raw+eip $(readlink -f $(which python3))` avoids
    needing root for every run).

Deduplicates exact-duplicate packets at capture time -- a real, confirmed
Linux kernel behavior, not a hypothetical: raw AF_PACKET capture on "lo"
delivers every packet TWICE (once via the loopback pseudo-device's egress
hook, once via its ingress hook), byte-for-byte identical, microseconds
apart. Confirmed directly on a real captured pcap from this project's own
testing: exactly half of 305 captured packets (152) were exact duplicates
of the immediately preceding packet, all within 1-14 microseconds -- far
too fast to be a genuine TCP retransmission (which needs at least one RTO
timer, tens of milliseconds minimum). Since this project's own loopback
auto-detection (below) makes "lo" the default capture interface for any
same-machine test, every downstream artifact (packet counts, byte counts,
burst/flow statistics, the handcrafted feature set) would otherwise
silently double-count real traffic. Filtered once, here, at the actual
capture source, rather than in every individual downstream consumer.
"""
try:
    from scapy.all import AsyncSniffer, get_if_list, wrpcap
    from scapy.error import Scapy_Exception

    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

import ipaddress
import socket
import time


class CaptureUnavailableError(RuntimeError):
    pass


def _is_loopback_host(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _check_raw_socket_permission():
    # scapy's own AsyncSniffer.start() only spawns a background thread and
    # returns immediately -- the actual raw-socket open happens inside that
    # thread (scapy's sendrecv.py _run()), so a real PermissionError there
    # never reaches this project's own try/except PermissionError in
    # start() below. Confirmed directly: running without root/CAP_NET_RAW
    # crashed that thread with a raw, unhandled "Exception in thread
    # AsyncSniffer" traceback instead of this project's own clear
    # CaptureUnavailableError -- and worse, the experiment silently
    # continued and "succeeded" with 0 packets captured, which defeats the
    # whole point of a traffic-fingerprinting testbed. Doing the identical
    # socket-open synchronously here, before scapy ever starts its thread,
    # surfaces the same real permission failure where it can actually be
    # caught. Only meaningful on Linux, where scapy's default backend opens
    # a raw AF_PACKET socket (confirmed: conf.use_pcap=False on this
    # project's dev machine) -- Windows/Npcap and macOS's BPF device use a
    # different permission model this check doesn't cover.
    if not hasattr(socket, "AF_PACKET"):
        return
    try:
        raw = socket.socket(socket.AF_PACKET, socket.SOCK_RAW)
        raw.close()
    except PermissionError as exc:
        raise CaptureUnavailableError(
            "Packet capture needs elevated privileges: run as Administrator "
            "(Windows/Npcap) or with sudo / CAP_NET_RAW (Linux/macOS)."
        ) from exc


class ScapyCapture:
    def __init__(self, host, port, output_path, interface=None, logger=None):
        self.host = host
        self.port = port
        self.output_path = output_path
        self.interface = interface
        self.logger = logger
        self._sniffer = None
        self._packets = []
        self._last_raw = None
        self._last_ts = None
        self._duplicate_count = 0
        # Public, read after stop() -- e.g. telemetry/manifest.py's
        # capture provenance block.
        self.capture_start_epoch = None
        self.capture_end_epoch = None
        self.written_packet_count = 0

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)

    def _on_packet(self, pkt):
        # See this module's own docstring: "lo" delivers every packet
        # twice, byte-for-byte identical, microseconds apart. A real
        # retransmission needs at least one RTO timer (tens of ms), so
        # comparing only against the immediately preceding packet with a
        # generous 2ms window safely catches the capture-level duplicate
        # without ever mistaking a genuine retransmission for one.
        raw = bytes(pkt)
        ts = float(pkt.time)
        if self._last_raw == raw and self._last_ts is not None and (ts - self._last_ts) < 0.002:
            self._duplicate_count += 1
            return
        self._last_raw = raw
        self._last_ts = ts
        self._packets.append(pkt)

    def start(self):
        if not SCAPY_AVAILABLE:
            raise CaptureUnavailableError(
                "scapy is not installed. Run `pip install scapy` and, on Windows, "
                "install Npcap (https://npcap.com/) first."
            )
        _check_raw_socket_permission()
        if self.interface is None and _is_loopback_host(self.host):
            # Real, confirmed bug, not a hypothetical: scapy's own default
            # interface selection (conf.iface) picks the machine's primary
            # NIC (e.g. a WiFi/ethernet adapter), never "lo" -- confirmed
            # directly on this project's dev machine (conf.iface reported
            # the WiFi adapter, with get_if_list() separately listing "lo").
            # Client/server traffic to a loopback host (127.0.0.1/::1/
            # localhost -- the exact setup main.py --interactive's own
            # guided setup and this project's own testing throughout use)
            # never touches that NIC at all -- it stays entirely inside the
            # kernel's loopback interface. Sniffing on the wrong interface
            # doesn't error, it just silently captures 0 packets forever,
            # defeating the whole point of a traffic-fingerprinting
            # testbed. core/config.py's capture_interface exists to let
            # someone override this, but main.py --interactive never
            # prompts for it, so it's always None in the common case this
            # fixes automatically here.
            self.interface = "lo"
            self._log("Host is loopback -- auto-selecting capture interface 'lo'")
        bpf_filter = f"tcp port {self.port}"
        self._log(f"Starting capture: filter='{bpf_filter}' interface={self.interface or 'default'}")
        try:
            self._sniffer = AsyncSniffer(
                filter=bpf_filter,
                iface=self.interface,
                prn=self._on_packet,
                store=False,
            )
            self._sniffer.start()
            self.capture_start_epoch = time.time()
        except PermissionError as exc:
            raise CaptureUnavailableError(
                "Packet capture needs elevated privileges: run as Administrator "
                "(Windows/Npcap) or with sudo / CAP_NET_RAW (Linux/macOS)."
            ) from exc

    def stop(self):
        if self._sniffer is None:
            return
        try:
            self._sniffer.stop()
        except (Scapy_Exception, AttributeError) as exc:
            # Real, confirmed scapy behavior on this platform, not a bug in
            # this project: with conf.use_pcap=False, scapy's AsyncSniffer
            # falls back to a native-Linux AF_PACKET socket whose sniff
            # thread never assigns self.stop_cb. Confirmed directly by
            # reading both scapy versions actually in play here: apt's
            # python3-scapy 2.5.0 wraps the resulting AttributeError into
            # its own Scapy_Exception("Unsupported (offline or unsupported
            # socket)"); this project's own pip-installed scapy 2.7.0 calls
            # self.stop_cb() with no try/except at all, so the identical
            # condition surfaces as a bare AttributeError instead -- same
            # root cause, two different exception types depending on which
            # scapy is on PATH, so both are caught here. Reproduces
            # regardless of root/CAP_NET_RAW. The sniff thread (scapy sets
            # thread.daemon=True) keeps running harmlessly in the
            # background rather than blocking process exit; whatever
            # packets it already delivered to _packets via prn= are still
            # real and worth keeping, so don't let this crash the whole
            # experiment and lose ground_truth.json over a capture-teardown
            # quirk unrelated to the actual workload that just ran.
            self._log(f"Capture stop() raised a known scapy issue, continuing: {exc}")
        self._sniffer = None
        self.capture_end_epoch = time.time()
        if self._duplicate_count:
            self._log(
                f"Dropped {self._duplicate_count} exact-duplicate packets "
                "(known Linux 'lo' double-capture behavior, see this module's docstring)."
            )
        self.written_packet_count = len(self._packets)
        if self._packets:
            wrpcap(str(self.output_path), self._packets)
            self._log(f"Wrote {len(self._packets)} packets to {self.output_path}")
        else:
            self._log("Capture stopped with 0 packets captured.")


def list_interfaces():
    if not SCAPY_AVAILABLE:
        raise CaptureUnavailableError("scapy is not installed.")
    return get_if_list()
