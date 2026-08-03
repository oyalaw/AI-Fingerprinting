"""Device registry: Level 5 of the fingerprinting taxonomy.

Devices aren't code plugins (no per-device adapter module to import), just
metadata describing platform family, capture capability, and whether the
device can run the full Python stack or only acts as a thin network client.
"""
from core.registry import DEVICES

# (name, platform_family, thin_client, capture_capable, capture_backend, notes)
_DEVICES = [
    ("windows", "windows", False, True, "npcap", "Requires Npcap + Administrator for capture."),
    ("ubuntu", "linux", False, True, "libpcap", "Requires libpcap + root/CAP_NET_RAW for capture."),
    ("raspberry_pi_5", "linux", False, True, "libpcap", "ARM64, CPU-only accelerator."),
    ("jetson_agx_orin", "linux", False, True, "libpcap", "ARM64, CUDA + TensorRT capable."),
    ("jetson_orin_nano", "linux", False, True, "libpcap", "ARM64, CUDA + TensorRT capable."),
    ("jetson_xavier", "linux", False, True, "libpcap", "ARM64, CUDA + TensorRT capable."),
    ("macbook", "macos", False, True, "libpcap", "Apple Silicon, MPS/CoreML capable."),
    ("samsung_galaxy", "android", True, False, None, "Thin network client; no root/raw sockets."),
    ("google_pixel", "android", True, False, None, "Thin network client; no root/raw sockets."),
    ("iphone", "ios", True, False, None, "Thin network client; sandboxed, no raw sockets."),
    ("ipad", "ipados", True, False, None, "Thin network client; sandboxed, no raw sockets."),
]

for _name, _platform, _thin, _capture, _backend, _notes in _DEVICES:
    DEVICES.add(
        _name,
        implemented=True,
        platform_family=_platform,
        thin_client=_thin,
        capture_capable=_capture,
        capture_backend=_backend,
        notes=_notes,
    )
