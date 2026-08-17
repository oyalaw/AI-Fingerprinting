"""ColossalAI -- registered but not yet implemented. Re-investigated on
Ubuntu (this project's original findings were all from a Windows dev
machine): the original Windows platform gate is confirmed gone, but a
different, real wall was found directly underneath it.

The original finding was `pip install colossalai` failing immediately with
`RuntimeError: Windows is not supported yet. Please try again within the
Windows Subsystem for Linux (WSL).` -- confirmed directly in ColossalAI
0.5.0's own `setup.py`: `if sys.platform == "win32": raise RuntimeError(...)`,
a deliberate maintainer gate, not a missing-wheel gap. On Ubuntu that check
never fires, and `pip install colossalai` genuinely succeeds -- confirmed
directly, `import colossalai` works (`colossalai.__version__ == "0.5.0"`).

But installing it is destructive to this project's own environment:
ColossalAI 0.5.0's `requirements/requirements.txt` pins `torch>=2.2.0,
<=2.5.1`, `transformers==4.51.3`, `diffusers==0.29.0`, and `uvicorn==0.29.0`
exactly -- incompatible with the versions this project's own already-verified
adapters depend on (torch 2.13.0, transformers 5.15.0, diffusers 0.39.0, and
flwr's own `uvicorn[standard]<0.50.0,>=0.49.0` requirement). Confirmed
directly: running the install downgraded all four in place, silently pulled
in full CUDA 12.4 `nvidia-*-cu12` wheels despite this being a CPU-only
machine (ColossalAI's own dependency resolution doesn't respect the
CPU-only torch index this project's other adapters were installed from),
and left `torchvision 0.28.0+cpu`/`flwr 1.33.0` in a broken, incompatible
state pointed at by pip's own resolver warning. Reverted immediately
(reinstalled torch==2.13.0/torchvision==0.28.0 from the CPU wheel index,
confirmed no lasting side effects). ColossalAI also pulls in a surprisingly
wide, unrelated dependency surface for a distributed-training package --
`ray`, `bitsandbytes`, `peft`, `galore_torch`, `fabric`, `paramiko`, `rpyc`
-- none of which a `torch.distributed`-style adapter should need.

Same class of finding as this project's other "layer resolved, deeper wall
found" investigations (e.g. distributed_frameworks/horovod_adapter.py) --
not fixed by switching to Linux, since the real blocker was never the
platform gate itself, and this stays a stub. Revisit only in a dedicated
venv isolated from this project's own PyTorch/transformers/diffusers/flwr
version requirements, not this shared environment.
"""
from core.registry import DISTRIBUTED_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError("ColossalAI is not yet implemented -- see this module's docstring.")


DISTRIBUTED_FRAMEWORKS.register("ColossalAI", implemented=False, organization="HPC AI Tech")(build)
