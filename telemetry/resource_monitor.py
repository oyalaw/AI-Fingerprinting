"""Resource telemetry: continuous CPU/memory/network/GPU/power/energy
sampling during an experiment, written to a background CSV while
role.run() is executing -- see core/experiment.py.

Deliberately a separate artifact from the attacker-facing traffic
features (traffic/*.py), not merged into them: if this project's threat
model is a passive network observer, host-side signals like GPU
utilization or CPU power must never leak into a fingerprinting
classifier's input features -- they're ground-truth/explanatory data for
understanding *why* workloads differ, not something a network observer
could ever see. Keeping them in a separate _resource.csv makes that
separation structurally hard to get wrong by accident.

Collectors are auto-detected per machine and skipped gracefully when
unavailable (no NVIDIA GPU, no RAPL read access, not running on a
Jetson) -- the CSV always has the same full column set regardless of
which collectors are actually active on a given machine, so cross-device
analysis doesn't need to handle a different schema per device; fields a
machine can't provide are just left blank for every row.

Power vs. energy, and total vs. delta, are deliberately not the same
column: bytes_sent_total/bytes_received_total are the OS's cumulative
interface counters (since boot), bytes_sent_delta/bytes_received_delta
are this-sample-minus-previous-sample (what you actually want for
per-interval analysis); *_power_w is instantaneous wattage, *_energy_j is
that power numerically integrated over time (E += P * dt) UNLESS a real
hardware energy counter is available (RAPL), which is used directly
instead of being estimated.
"""
import csv
import datetime
import pathlib
import re
import subprocess
import threading
import time

FIELDNAMES = [
    "experiment_id",
    "timestamp_utc",
    "relative_time_sec",
    "sample_index",
    "role",
    "device",
    "sample_interval_ms",
    "telemetry_source",
    "bytes_sent_total",
    "bytes_received_total",
    "bytes_sent_delta",
    "bytes_received_delta",
    "cpu_usage_percent",
    "memory_usage_mb",
    "memory_usage_percent",
    "gpu_usage_percent",
    "gpu_memory_used_mb",
    "gpu_memory_total_mb",
    "cpu_power_w",
    "gpu_power_w",
    "system_power_w",
    "cpu_energy_j",
    "gpu_energy_j",
    "system_energy_j",
]


class _CPUMemoryNetworkCollector:
    """psutil-based CPU/memory/network sampling -- psutil is already a
    real dependency of this project (confirmed importable), available on
    every platform this project targets, so this collector is never
    expected to be unavailable."""

    name = "psutil"

    def __init__(self):
        import psutil

        self._psutil = psutil
        # cpu_percent()'s first call always returns a meaningless 0.0 --
        # it measures usage *since the last call*, so prime it here rather
        # than let the very first real sample be garbage.
        psutil.cpu_percent(interval=None)

    def sample(self):
        psutil = self._psutil
        vm = psutil.virtual_memory()
        net = psutil.net_io_counters()
        return {
            "cpu_usage_percent": psutil.cpu_percent(interval=None),
            "memory_usage_mb": vm.used / (1024 * 1024),
            "memory_usage_percent": vm.percent,
            "bytes_sent_total": net.bytes_sent,
            "bytes_received_total": net.bytes_recv,
        }

    def close(self):
        pass


class _NvidiaGPUCollector:
    """NVML-based GPU utilization/memory/power, via the pynvml package
    already in this project's dependency set. Works for a real discrete
    NVIDIA GPU; on Jetson boards NVML support varies by JetPack release
    (confirmed elsewhere in this project's own history: this machine's
    own GPU-less run and the Jetson's earlier CUDA investigation both
    needed real, direct checks rather than assuming -- nvmlInit() itself
    is the check here, and this collector simply won't register at all
    if it fails, letting _JetsonTegrastatsCollector below cover GPU%/
    power on boards where NVML doesn't)."""

    name = "nvml"

    def __init__(self):
        import pynvml

        pynvml.nvmlInit()
        self._pynvml = pynvml
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def sample(self):
        pynvml = self._pynvml
        result = {}
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            result["gpu_usage_percent"] = float(util.gpu)
        except pynvml.NVMLError:
            pass
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            result["gpu_memory_used_mb"] = mem.used / (1024 * 1024)
            result["gpu_memory_total_mb"] = mem.total / (1024 * 1024)
        except pynvml.NVMLError:
            pass
        try:
            result["gpu_power_w"] = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
        except pynvml.NVMLError:
            pass  # power reporting isn't exposed on every board/driver combination
        return result

    def close(self):
        try:
            self._pynvml.nvmlShutdown()
        except Exception:
            pass


class _RAPLCollector:
    """Intel RAPL CPU energy, read directly from the kernel's own
    cumulative hardware energy counter (sysfs) -- preferred over
    numerical integration wherever available, since it's real hardware
    telemetry, not an estimate. Confirmed directly on this project's own
    dev machine: /sys/class/powercap/intel-rapl:0/energy_uj exists but is
    root-only (`-r--------`), so a normal user's read raises
    PermissionError immediately -- that's the real, expected case this
    collector's own try/except in _detect_collectors() below handles by
    just not registering, not a bug to work around here. To enable it:
    `sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj` (resets
    on reboot) or a persistent udev rule, at the user's own discretion --
    not done automatically by this project."""

    name = "rapl"
    _PATH = pathlib.Path("/sys/class/powercap/intel-rapl:0/energy_uj")

    def __init__(self):
        self._last_uj = int(self._PATH.read_text())
        self._last_t = time.monotonic()

    def sample(self):
        now_uj = int(self._PATH.read_text())
        now_t = time.monotonic()
        delta_uj = now_uj - self._last_uj
        delta_t = now_t - self._last_t
        self._last_uj, self._last_t = now_uj, now_t
        if delta_uj < 0 or delta_t <= 0:
            # RAPL's counter wraps around at a hardware-specific max value
            # -- documented behavior, not a bug in this collector. Just
            # skip this one sample rather than report a bogus negative
            # power reading.
            return {}
        return {
            "cpu_power_w": (delta_uj / 1_000_000.0) / delta_t,
            "cpu_energy_j": delta_uj / 1_000_000.0,
        }

    def close(self):
        pass


class _JetsonTegrastatsCollector:
    """GPU utilization and power rails via NVIDIA's own `tegrastats`
    binary, for Jetson boards where NVML (see _NvidiaGPUCollector) either
    isn't available or doesn't expose power. Parsing is based on this
    exact project's own real, directly-captured tegrastats output (from
    an actual Jetson used earlier in this project's development):

        RAM 5953/7620MB (lfb 2x4MB) SWAP 770/3810MB (cached 4MB)
        CPU [30%@1344,...] GR3D_FREQ 63% cpu@48.312C ...
        VDD_IN 6526mW/6526mW VDD_CPU_GPU_CV 1382mW/1382mW VDD_SOC 1697mW/1697mW

    Real, confirmed limitation, not a gap in this collector: on this
    exact tegrastats output, CPU and GPU power are only ever reported as
    one *combined* VDD_CPU_GPU_CV rail, never separately -- fabricating a
    cpu_power_w/gpu_power_w split from a single combined number would be
    reporting a precision this hardware doesn't actually provide, so this
    collector deliberately leaves both blank and only reports the real,
    unambiguous total from VDD_IN as system_power_w. Only registers at
    all when the `tegrastats` binary is actually present, so this stays
    completely inert on every non-Jetson machine.

    Not independently verified end-to-end against a live tegrastats
    process by this module's own author -- built directly from this
    project's own captured sample output above, not guessed from generic
    docs, but confirm the regexes still match on your specific JetPack
    release before trusting this collector's numbers.
    """

    name = "tegrastats"
    _GR3D_RE = re.compile(r"GR3D_FREQ (\d+)%")
    _VDD_IN_RE = re.compile(r"VDD_IN (\d+)mW/")

    def __init__(self, interval_ms):
        import shutil

        binary = shutil.which("tegrastats")
        if not binary:
            raise RuntimeError("tegrastats not found on PATH -- not a Jetson, or JetPack not installed.")
        self._proc = subprocess.Popen(
            [binary, "--interval", str(max(interval_ms, 100))],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def sample(self):
        line = self._proc.stdout.readline()
        if not line:
            return {}
        result = {}
        gr3d = self._GR3D_RE.search(line)
        if gr3d:
            result["gpu_usage_percent"] = float(gr3d.group(1))
        vdd_in = self._VDD_IN_RE.search(line)
        if vdd_in:
            result["system_power_w"] = int(vdd_in.group(1)) / 1000.0
        return result

    def close(self):
        self._proc.terminate()


def _detect_collectors(sample_interval_ms, logger=None):
    # psutil failing is a real setup problem, not expected hardware
    # variation -- it's a hard dependency of this project (confirmed
    # elsewhere: torch, scipy, etc. all assume it's present), so unlike
    # the other three (where "this machine has no NVIDIA GPU"/"no RAPL
    # access"/"not a Jetson" is the normal, silent case), a missing
    # psutil gets a loud warning instead of vanishing into the same
    # blanket except. Real bug found this way, not hypothetical: a
    # conda env used for actual experiment runs never had psutil
    # installed, so every resource_csv it produced silently had
    # telemetry_source="none" and every single field blank on every
    # row -- invisible unless someone thought to open the CSV and
    # notice the empty columns.
    collectors = []
    for name, build in (
        ("psutil", lambda: _CPUMemoryNetworkCollector()),
        ("nvml", lambda: _NvidiaGPUCollector()),
        ("rapl", lambda: _RAPLCollector()),
        ("tegrastats", lambda: _JetsonTegrastatsCollector(sample_interval_ms)),
    ):
        try:
            collectors.append(build())
        except Exception as exc:
            if name == "psutil" and logger:
                logger.warning(
                    f"Resource telemetry: psutil collector failed to initialize ({exc!r}) -- "
                    "this is this project's baseline collector and should always work; "
                    "resource_csv will be missing CPU/memory/network data. "
                    "Run `pip install psutil` in whichever environment is actually running this."
                )
            continue  # the other three genuinely just aren't available on every machine
    if not collectors and logger:
        logger.warning("Resource telemetry: no collectors available at all -- resource_csv rows will be empty.")
    return collectors


class ResourceMonitor:
    """Samples every available collector on a background thread every
    sample_interval_ms and appends one row per sample to output_csv.
    start()/stop() mirror traffic/packet_features.py's ScapyCapture
    lifecycle -- construct, start() before the workload runs, stop() in
    a finally block afterward (see core/experiment.py)."""

    def __init__(self, experiment_id, role, device, output_csv, sample_interval_ms=500, logger=None):
        self.experiment_id = experiment_id
        self.role = role
        self.device = device
        self.output_csv = pathlib.Path(output_csv)
        self.sample_interval_ms = sample_interval_ms
        self.logger = logger
        self._collectors = _detect_collectors(sample_interval_ms, logger=logger)
        self._sources = "+".join(c.name for c in self._collectors) or "none"
        if logger:
            logger.info(f"Resource telemetry collectors active: {self._sources}")
        self._thread = None
        self._stop_event = threading.Event()
        self._energy_totals = {"gpu_energy_j": 0.0, "system_energy_j": 0.0}
        self._prev_bytes = {}
        self._file = None
        self._writer = None

    def start(self):
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_csv.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDNAMES, extrasaction="ignore")
        self._writer.writeheader()
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        sample_index = 0
        prev_t = time.monotonic()
        while not self._stop_event.wait(self.sample_interval_ms / 1000.0):
            now = time.monotonic()
            dt = now - prev_t
            prev_t = now
            row = {
                "experiment_id": self.experiment_id,
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "relative_time_sec": now - self._start_time,
                "sample_index": sample_index,
                "role": self.role,
                "device": self.device,
                "sample_interval_ms": self.sample_interval_ms,
                "telemetry_source": self._sources,
            }
            for collector in self._collectors:
                try:
                    row.update(collector.sample())
                except Exception:
                    continue  # one bad sample from one collector shouldn't kill the whole monitor

            for key in ("bytes_sent", "bytes_received"):
                total = row.get(f"{key}_total")
                if total is not None:
                    prev = self._prev_bytes.get(key)
                    row[f"{key}_delta"] = (total - prev) if prev is not None else 0
                    self._prev_bytes[key] = total

            # Energy: RAPL already reports cpu_energy_j directly above (a
            # real hardware counter) -- everything else is numerically
            # integrated from instantaneous power, E += P * dt, since no
            # other collector here exposes a hardware energy counter.
            if row.get("gpu_power_w") is not None:
                self._energy_totals["gpu_energy_j"] += row["gpu_power_w"] * dt
                row["gpu_energy_j"] = self._energy_totals["gpu_energy_j"]
            system_power = row.get("system_power_w")
            if system_power is None:
                parts = [row.get("cpu_power_w"), row.get("gpu_power_w")]
                parts = [p for p in parts if p is not None]
                system_power = sum(parts) if parts else None
            if system_power is not None:
                row["system_power_w"] = system_power
                self._energy_totals["system_energy_j"] += system_power * dt
                row["system_energy_j"] = self._energy_totals["system_energy_j"]

            self._writer.writerow(row)
            self._file.flush()
            sample_index += 1

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=self.sample_interval_ms / 1000.0 + 2)
        for collector in self._collectors:
            collector.close()
        if self._file:
            self._file.close()
