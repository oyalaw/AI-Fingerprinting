"""FedScale -- deliberately left a stub after real investigation, further
along than most of this file's other findings.

Unlike fl_frameworks/fedjax_adapter.py, this one isn't a structural
mismatch: `pip install fedscale` succeeds cleanly (pure Python wheel, no
compiler needed), and `fedscale.cloud.channels` contains real
gRPC-generated code (`JobServiceServicer`/`JobServiceStub`, via
`job_api_pb2_grpc.py`) -- genuine networked client/server infrastructure,
the same shape as fl_frameworks/flower_adapter.py, not FedJAX's pure
in-process simulation.

Tried to go further and actually instantiate `fedscale.cloud.aggregation
.aggregator.Aggregator` (the server) to check real integration
complexity, and hit a chain of eager imports, each fixed in turn: missing
`wandb` (installed), missing `tensorboard` (installed), then
`fedscale.cloud.fllibs` unconditionally does
`from fedscale.utils.models.tensorflow_model_provider import
get_tensorflow_model`, which does `import tensorflow` at module level --
with no lazy-loading or PyTorch/TensorFlow branch to route around it,
unlike the SpeechBrain CRDNN situation
(speech_frameworks/speechbrain_adapter.py) where a direct submodule
import avoided the broken path entirely. This import happens
unconditionally even for a pure-PyTorch FedScale job. Checked directly:
plain `tensorflow` has no wheel at all for this Python version -- the
same root-cause blocker already documented on
frameworks/tensorflow_adapter.py itself. FedScale is blocked on that same
wall, one layer removed.

Revisit once frameworks/tensorflow_adapter.py has a real TensorFlow
install to build against -- unlike FedJAX, there's a real integration
worth returning to here, not a dead end in principle.

That revisit condition is now met: re-checked on Ubuntu after
frameworks/tensorflow_adapter.py's own wheel wall resolved (TensorFlow
2.21.0 ships a real `cp313` wheel -- see that module's docstring). With
`tensorflow`/`wandb`/`tensorboard` all installed, confirmed directly:
`from fedscale.cloud.aggregation.aggregator import Aggregator` now
imports cleanly, no error -- the entire chain this docstring previously
traced (`fllibs` -> `tensorflow_model_provider` -> `import tensorflow`)
is fully resolved. What's left isn't an environment blocker anymore: a
real adapter needs `Aggregator`/`Executor` wired to this project's
`FLFrameworkAdapter` interface, its own config-object API (FedScale
expects a populated `args` namespace covering client sampling, rounds,
model name, etc. -- a wider surface than Flower/FedLab's constructors),
and its gRPC `JobServiceServicer`/`JobServiceStub` job-submission protocol
mapped onto this project's client/server roles. That's a genuinely new
adapter to write, not another investigation pass -- left as a stub for
that reason, not because anything here still fails to install or import.
"""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "FedScale's TensorFlow import wall is resolved on Ubuntu -- what's left is "
        "writing a real Aggregator/Executor adapter, not an environment blocker -- "
        "see this module's docstring."
    )


FL_FRAMEWORKS.register("FedScale", implemented=False, organization="Michigan State University")(build)
