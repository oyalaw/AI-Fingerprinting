"""Operating system registry: Level 6 of the fingerprinting taxonomy --
split out from core/devices.py's own Level 5 (real hardware). See that
module's docstring for why: `ubuntu`/`windows` used to sit in the device
list as if they were hardware, which broke down once a single device
(a DGX) could plausibly run more than one OS.

`platform_tags` here is what main.py's `--interactive` flow unions with
core/devices.py's own (now hardware-only) `platform_tags` to filter the
Framework prompt -- e.g. TensorRT/DeepStream/Edge Impulse gate on the
device's `"jetson"` tag, while TensorFlow/PyTorch/etc. availability gates
on the OS's `"linux"`/`"windows"`/`"macos"`/etc. tag instead. No separate
`platform_family` field the way devices used to carry one -- after the
split, family and the single platform tag are always identical here, so
keeping both would be pure duplication.
"""
from core.registry import OPERATING_SYSTEMS

# (name, platform_tags, notes)
_OPERATING_SYSTEMS = [
    ("Windows", ("windows",), "Requires Npcap + Administrator for capture."),
    ("Ubuntu", ("linux",), "Requires libpcap + root/CAP_NET_RAW for capture."),
    ("macOS", ("macos",), "Requires libpcap + elevated privileges for capture."),
    (
        "JetPack",
        ("linux",),
        "NVIDIA's own Ubuntu-based L4T distribution for Jetson boards -- a real "
        "Linux for framework-compatibility purposes, not a distinct platform family "
        "(core/devices.py's own \"jetson\" tag on the Jetson boards is what actually "
        "gates TensorRT/DeepStream/Edge Impulse, a hardware fact, not an OS one).",
    ),
    ("Android", ("android",), "Thin network client; no root/raw sockets."),
    ("iOS", ("ios",), "Thin network client; sandboxed, no raw sockets."),
    ("iPadOS", ("ipados",), "Thin network client; sandboxed, no raw sockets."),
]

for _name, _tags, _notes in _OPERATING_SYSTEMS:
    OPERATING_SYSTEMS.add(_name, implemented=True, platform_tags=_tags, notes=_notes)
