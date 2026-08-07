"""LMDeploy -- deliberately left a stub after real investigation.

Checked directly: `pip install lmdeploy` fails dependency resolution --
it hard-requires `ray`, which has no wheel at all for this Python version
in this environment. Same root cause already documented on
distributed_frameworks/ray_train_adapter.py; nothing specific to
LMDeploy itself to add.
"""
from core.registry import LLM_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "LMDeploy requires ray, which has no wheel for this Python version -- see "
        "distributed_frameworks/ray_train_adapter.py's docstring for the same finding."
    )


LLM_FRAMEWORKS.register("LMDeploy", implemented=False, use_case="Efficient deployment")(build)
