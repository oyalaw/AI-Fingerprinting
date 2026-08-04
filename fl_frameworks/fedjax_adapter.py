"""FedJAX -- deliberately left a stub, not a gap, same standing as
frameworks/metal_performance_shaders_adapter.py.

Two separate reasons, both checked directly against the installed
package (`pip install fedjax` succeeds cleanly, `import fedjax` works
fine on this Windows machine -- this isn't a wheel/platform problem):

1. No PyTorch bridge. Every other framework/FL adapter in this project
   reuses the exact same PyTorch ResNet18 module, converted or wrapped
   into that framework's native format. FedJAX has no such path --
   models must be built natively as JAX/Haiku (`fedjax.create_model_from_haiku`)
   or Stax (`fedjax.create_model_from_stax`) modules, trained via JAX's
   own autodiff (`fedjax.grad`). A real FedJAX adapter would need an
   entirely separate, hand-written Haiku CNN, not this project's shared
   architecture.

2. No network transport at all, at any level. FedJAX's full top-level API
   (`fedjax.InMemoryFederatedData`, `fedjax.for_each_client`,
   `fedjax.FederatedAlgorithm`, `fedjax.algorithms`, ...) has no
   server/client role split, no host/port/socket parameter anywhere in
   the entire surface. It's a single-process research/benchmarking
   library for evaluating FL *algorithms* against in-memory datasets, not
   a deployment framework. This is a stronger version of the problem
   fl_frameworks/nvflare_adapter.py's simulator_run() has: NVFlare at
   least offers a real multi-process POC/production mode as an
   alternative; FedJAX has no such option at all, in any mode. There is
   no way to make this adapter generate real, capturable network traffic
   -- not "not implemented yet," structurally impossible for what this
   project needs.

If FedJAX ever grows a real distributed deployment mode, or reusing this
project's PyTorch model via JAX's PyTorch interop becomes practical,
revisit this. Until then it stays a stub on principle.
"""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "FedJAX has no network transport and no PyTorch model bridge -- "
        "see this module's docstring."
    )


FL_FRAMEWORKS.register("FedJAX", implemented=False, organization="Google")(build)
