"""Device registry: Level 5 of the fingerprinting taxonomy.

Devices aren't code plugins (no per-device adapter module to import), just
metadata describing platform family, capture capability, and whether the
device can run the full Python stack or only acts as a thin network client.

`platform_tags` is the set of tags this device satisfies for matching
against a framework's own `platforms=[...]` registration metadata (see
frameworks/tensorrt_adapter.py etc.) -- used by main.py's `--interactive`
flow to only offer frameworks the selected device can actually run. It's
usually just `(platform_family,)`, but the three Jetson boards need the
extra "jetson" tag: their `platform_family` is "linux" (they run the same
Python stack as Ubuntu), but several frameworks (TensorRT, DeepStream,
Edge Impulse) register `platforms=["jetson"]` specifically rather than the
broader "linux", since they need Jetson's CUDA/TensorRT stack, not just
any Linux box.
"""
from core.registry import DEVICES

# (name, platform_family, platform_tags, thin_client, capture_capable, capture_backend, notes)
_DEVICES = [
    ("windows", "windows", ("windows",), False, True, "npcap", "Requires Npcap + Administrator for capture."),
    ("ubuntu", "linux", ("linux",), False, True, "libpcap", "Requires libpcap + root/CAP_NET_RAW for capture."),
    ("raspberry_pi_5", "linux", ("linux",), False, True, "libpcap", "ARM64, CPU-only accelerator."),
    ("jetson_agx_orin", "linux", ("linux", "jetson"), False, True, "libpcap", "ARM64, CUDA + TensorRT capable."),
    ("jetson_orin_nano", "linux", ("linux", "jetson"), False, True, "libpcap", "ARM64, CUDA + TensorRT capable."),
    ("jetson_xavier", "linux", ("linux", "jetson"), False, True, "libpcap", "ARM64, CUDA + TensorRT capable."),
    ("macbook", "macos", ("macos",), False, True, "libpcap", "Apple Silicon, MPS/CoreML capable."),
    ("samsung_galaxy", "android", ("android",), True, False, None, "Thin network client; no root/raw sockets."),
    ("google_pixel", "android", ("android",), True, False, None, "Thin network client; no root/raw sockets."),
    ("iphone", "ios", ("ios",), True, False, None, "Thin network client; sandboxed, no raw sockets."),
    ("ipad", "ipados", ("ipados",), True, False, None, "Thin network client; sandboxed, no raw sockets."),
]

for _name, _platform, _tags, _thin, _capture, _backend, _notes in _DEVICES:
    DEVICES.add(
        _name,
        implemented=True,
        platform_family=_platform,
        platform_tags=_tags,
        thin_client=_thin,
        capture_capable=_capture,
        capture_backend=_backend,
        notes=_notes,
    )
