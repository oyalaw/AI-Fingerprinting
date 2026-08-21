"""Device registry: Level 5 of the fingerprinting taxonomy -- real physical
hardware only. Operating system is a separate axis (core/operating_systems.py,
Level 6) -- these two used to be conflated in one list here (`ubuntu`/
`windows` sitting alongside real hardware like `iphone`/`jetson_agx_orin`
as if they were the same kind of thing), which broke down the moment a
device could plausibly run more than one OS (a DGX runs a Ubuntu-based OS;
there was no way to say "DGX hardware, running Ubuntu" without `ubuntu`
and `dgx` colliding as two unrelated same-level entries).

`platform_tags` here is now hardware-only -- only the three Jetson boards
have a non-empty one (`("jetson",)`, the real CUDA/TensorRT-silicon tag
TensorRT/DeepStream/Edge Impulse's own `platforms=["jetson"]` registrations
match against). Every other device's framework compatibility is entirely
OS-driven (TensorFlow/PyTorch/etc. availability doesn't care which x86_64
box or which ARM64 SBC you're on, only which OS it's running), so their
`platform_tags` is empty -- that signal now lives on
core/operating_systems.py's own entries instead.

`compatible_os` is a tuple of operating_system names this device can
realistically run -- used by main.py's `--interactive` flow to narrow the
OS prompt to what's actually possible for the device just picked, the same
way `platform_tags` narrows the Framework prompt.
"""
from core.registry import DEVICES

# (name, platform_tags, thin_client, capture_capable, capture_backend, compatible_os, notes)
_DEVICES = [
    ("pc", (), False, True, "libpcap", ("Windows", "Ubuntu"), "Generic x86_64 desktop/laptop, no dedicated GPU."),
    ("dgx", (), False, True, "libpcap", ("Ubuntu",), "x86_64, multi-GPU (real CUDA/TensorRT, NVLink)."),
    ("raspberry_pi_5", (), False, True, "libpcap", ("Ubuntu",), "ARM64, CPU-only accelerator."),
    ("jetson_agx_orin", ("jetson",), False, True, "libpcap", ("JetPack",), "ARM64, CUDA + TensorRT capable."),
    ("jetson_orin_nano", ("jetson",), False, True, "libpcap", ("JetPack",), "ARM64, CUDA + TensorRT capable."),
    ("jetson_xavier", ("jetson",), False, True, "libpcap", ("JetPack",), "ARM64, CUDA + TensorRT capable."),
    ("macbook", (), False, True, "libpcap", ("macOS",), "Apple Silicon, MPS/CoreML capable."),
    ("samsung_galaxy", (), True, False, None, ("Android",), "Thin network client; no root/raw sockets."),
    ("google_pixel", (), True, False, None, ("Android",), "Thin network client; no root/raw sockets."),
    ("iphone", (), True, False, None, ("iOS",), "Thin network client; sandboxed, no raw sockets."),
    ("ipad", (), True, False, None, ("iPadOS",), "Thin network client; sandboxed, no raw sockets."),
]

for _name, _tags, _thin, _capture, _backend, _compatible_os, _notes in _DEVICES:
    DEVICES.add(
        _name,
        implemented=True,
        platform_tags=_tags,
        thin_client=_thin,
        capture_capable=_capture,
        capture_backend=_backend,
        compatible_os=_compatible_os,
        notes=_notes,
    )
