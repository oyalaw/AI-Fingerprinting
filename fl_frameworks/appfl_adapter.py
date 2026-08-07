"""APPFL -- registered but not yet implemented. Got much further after
installing a real C/C++ compiler (Visual Studio Build Tools, installed
specifically to unblock this and several other stubs sharing the "no
compiler" root cause) -- and it does bottom out in a real wall, just a
different and more specific one than before.

`pip install appfl` now succeeds completely (the original numpy-build
finding is resolved). `import appfl` shows real, genuine gRPC-based
client/server infrastructure (`appfl.run_grpc_server`/`run_grpc_client`,
`appfl.comm.grpc`) -- the same networked shape as
fl_frameworks/flower_adapter.py and fl_frameworks/fedscale_adapter.py,
not FedJAX's in-process-only situation.

But actually importing the gRPC entrypoint hits a real bug in APPFL's own
dependency version compatibility: `appfl.comm.grpc`'s package `__init__.py`
unconditionally imports its auth layer (`from .serve import serve` ->
`.auth` -> `appfl.login_manager` -> `.globus` -> `.tokenstore`), which
does `from globus_sdk.tokenstorage import SQLiteAdapter` -- but the
globus_sdk version that gets auto-installed (4.8.1) has reorganized that
API and no longer has a `tokenstorage` submodule at that path. Confirmed
there's no lazy-loading escape hatch the way
speech_frameworks/speechbrain_adapter.py found for SpeechBrain's CRDNN
lobe -- tried importing a lower-level class directly
(`appfl.comm.grpc.grpc_communicator.GRPCCommunicator`) and hit the
identical chain, since the package's own `__init__.py` runs unconditionally
on any import from that subpackage. A real APPFL/globus_sdk version
mismatch, not an environment gap -- revisit if APPFL pins an older
globus_sdk or updates its own Globus integration for the new API.
"""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "APPFL's gRPC server/client entrypoint unconditionally imports its Globus "
        "auth layer, which needs a globus_sdk API removed in the version that gets "
        "installed -- see this module's docstring."
    )


FL_FRAMEWORKS.register("APPFL", implemented=False, organization="Argonne National Laboratory")(build)
