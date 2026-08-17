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

Investigated exactly how wide that "own config-object API" surface is,
rather than leaving that vague. `Aggregator.__init__(self, args)` takes a
single `args` namespace -- confirmed directly by loading FedScale's own
default (`fedscale.cloud.config_parser.args`): it's not a handful of
fields, it's roughly 90, and they reveal what FedScale actually is --
not a library to call, but a full cloud/mobile-device-simulation
*platform* with its own launcher model. `executor_configs` (a string like
`'127.0.0.1:[1]'` naming IP:core-count pairs), `ps_ip`/`ps_port` (its own
parameter-server addressing, separate from this project's own `host`/
`port`), `device_avail_file`/`device_conf_file` (simulated per-device
availability traces), `model_zoo: 'torchcv'` + `model: 'shufflenet_v2_x2_0'`
(models selected from FedScale's own zoo by name, not handed a real
`nn.Module` directly), and `wandb_token` (logging assumed mandatory) are
all representative, not edge cases. Bridging this onto this project's
simple two-role (`--role server`/`--role client`) architecture -- which
every other FL adapter here does directly against a constructor taking a
handful of arguments -- would mean either faking out most of that
90-field surface with placeholder values (fragile, likely to break on
whatever FedScale code path actually reads them) or building real
adapter-side support for FedScale's own device-simulation/executor
launcher concept, which this project has no equivalent of anywhere else.
Concretely larger than distributed_frameworks/ray_train_adapter.py's or
frameworks/tvm_adapter.py's rewrites (each a few hours against a stable,
single-purpose API). Left as a stub for that reason, tracked here with
the concrete scope rather than a vague "needs more work."
"""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "FedScale's TensorFlow import wall is resolved on Ubuntu -- what's left is "
        "writing a real Aggregator/Executor adapter, not an environment blocker -- "
        "see this module's docstring."
    )


FL_FRAMEWORKS.register("FedScale", implemented=False, organization="Michigan State University")(build)
