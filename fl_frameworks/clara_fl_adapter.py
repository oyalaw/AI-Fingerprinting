"""Clara FL -- deliberately left a stub, for a reason distinct from the
usual install/build failures elsewhere in this file: it doesn't really
exist as a separate thing to install anymore.

Checked directly: `pip install clara-train-sdk` returns "No matching
distribution found" -- unlike fl_frameworks/leaf_adapter.py's plain
"never published" situation, this one has a specific reason: NVIDIA
folded Clara Train's federated learning capability into NVFlare (the
open-sourced continuation) starting around 2021-2022. This project's
fl_frameworks/nvflare_adapter.py is already implemented=True -- for
practical purposes that already IS "Clara FL", under its current name.
No separate adapter needed or possible; this entry stays a stub only
because it's a distinct name in the taxonomy, not because there's
unfinished work.
"""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "Clara FL was folded into NVFlare (fl_frameworks/nvflare_adapter.py, "
        "already implemented) -- see this module's docstring."
    )


FL_FRAMEWORKS.register("Clara FL", implemented=False, organization="NVIDIA")(build)
