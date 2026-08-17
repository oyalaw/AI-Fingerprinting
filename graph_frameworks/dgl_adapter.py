"""DGL -- deliberately left a stub after real investigation, not just an
unattempted placeholder.

Checked directly: PyPI's `dgl` package is stuck at version 0.1.3, an
alpha-status snapshot from early in the project's history (2018-era) --
confirmed by downloading and inspecting the wheel itself (411KB, `Deep
Graph Library` / `Development Status :: 3 - Alpha` in its own metadata),
abandoned on PyPI once the project moved to distributing current releases
through its own wheel index instead. That snapshot's API (dgl.graph.DGLGraph,
no GraphConv layer, no heterograph support) predates essentially everything
"DGL" means in current usage/tutorials, so using it would be misleading
under the DGL label even though it's technically the real project's own
code.

Tried the real, current path too: `pip install dgl -f
https://data.dgl.ai/wheels/torch-2.4/repo.html` (DGL's own documented
install instructions) -- no wheel matches this environment (Windows +
Python 3.14), so it falls back to the same stale 2018 snapshot from PyPI.
Modern DGL wheels are Linux-focused and pinned to specific supported
Python versions; there's currently no way to get real current DGL running
here at all, not even at the conversion-only tier frameworks/coreml_adapter.py
or frameworks/tensorrt_adapter.py manage.

If this project runs on Linux with an older, DGL-supported Python version,
this is worth revisiting: DGL's GraphConv is a genuine second GCN
implementation to pair with the existing GCN/Karate Club slice, the same
role graph_frameworks/pytorch_geometric_adapter.py already fills.

This project has now moved to exactly that: a real Linux (Ubuntu 24.04)
machine. Re-checked directly, and this specific wall doesn't move --
`pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/repo.html` on
Ubuntu still falls back to the identical stale `dgl-0.1.3` 2018 snapshot.
Confirmed why: DGL's own wheel index for that torch build lists real
current releases (up through 2.5.0) for `cp38` through `cp312` only --
still no `cp313` wheel at all, on Linux or otherwise. So "Linux" alone
wasn't the missing piece; being on Linux with a DGL-supported Python
version is, and this project's Python (3.13) is newer than DGL's current
wheel ceiling regardless of platform. Revisit once DGL publishes a cp313
wheel, or if running this project on an older, DGL-supported Python
becomes acceptable.
"""
from core.registry import GRAPH_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("DGL is not usable in this environment -- see this module's docstring.")


GRAPH_FRAMEWORKS.register("DGL", implemented=False, organization="AWS/DMLC")(build)
