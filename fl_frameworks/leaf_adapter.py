"""LEAF -- deliberately left a stub, not a gap: never published as a pip
package at all.

Checked directly: `pip install leaf-fl` returns "No matching distribution
found" -- no wheel or sdist under this or related names. Consistent with
LEAF's real nature: it's a benchmark suite (standardized non-IID dataset
partitions + reference task definitions for FL research, from Caldas et
al. 2018), distributed as a GitHub repo of data-preparation scripts, not
an installable library with a training/serving API the way Flower or
FedScale are. There's no adapter shape that would make sense here even
setting pip installability aside -- LEAF is something other frameworks'
experiments are built *on top of* (e.g. as a data source), not itself a
federated learning engine to run.
"""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "LEAF has no pip package and isn't itself an FL engine -- see this "
        "module's docstring."
    )


FL_FRAMEWORKS.register("LEAF", implemented=False, organization="Carnegie Mellon University")(build)
