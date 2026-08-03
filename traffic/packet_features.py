"""scapy-based background packet capture, scoped to one experiment's
host:port. This is the primary data source for the whole project: everything
downstream (flow/burst features, sequence export) reads the .pcap this writes.

Requires a working packet-capture driver:
  - Windows: Npcap (https://npcap.com/), running as Administrator.
  - Linux/macOS: libpcap (usually preinstalled) + root or CAP_NET_RAW
    (`sudo setcap cap_net_raw+eip $(readlink -f $(which python3))` avoids
    needing root for every run).
"""
try:
    from scapy.all import AsyncSniffer, get_if_list, wrpcap

    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False


class CaptureUnavailableError(RuntimeError):
    pass


class ScapyCapture:
    def __init__(self, host, port, output_path, interface=None, logger=None):
        self.host = host
        self.port = port
        self.output_path = output_path
        self.interface = interface
        self.logger = logger
        self._sniffer = None
        self._packets = []

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)

    def start(self):
        if not SCAPY_AVAILABLE:
            raise CaptureUnavailableError(
                "scapy is not installed. Run `pip install scapy` and, on Windows, "
                "install Npcap (https://npcap.com/) first."
            )
        bpf_filter = f"tcp port {self.port}"
        self._log(f"Starting capture: filter='{bpf_filter}' interface={self.interface or 'default'}")
        try:
            self._sniffer = AsyncSniffer(
                filter=bpf_filter,
                iface=self.interface,
                prn=self._packets.append,
                store=False,
            )
            self._sniffer.start()
        except PermissionError as exc:
            raise CaptureUnavailableError(
                "Packet capture needs elevated privileges: run as Administrator "
                "(Windows/Npcap) or with sudo / CAP_NET_RAW (Linux/macOS)."
            ) from exc

    def stop(self):
        if self._sniffer is None:
            return
        self._sniffer.stop()
        self._sniffer = None
        if self._packets:
            wrpcap(str(self.output_path), self._packets)
            self._log(f"Wrote {len(self._packets)} packets to {self.output_path}")
        else:
            self._log("Capture stopped with 0 packets captured.")


def list_interfaces():
    if not SCAPY_AVAILABLE:
        raise CaptureUnavailableError("scapy is not installed.")
    return get_if_list()
