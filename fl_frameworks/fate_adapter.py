"""FATE -- deliberately left a stub, for two independent real reasons.

Checked directly: `pip install fate-client` succeeds cleanly. But
`fate-client` (`fate_client.flow_cli`/`flow_sdk`/`pipeline`) is a remote
job-submission SDK for an already-deployed FATE cluster (the equivalent
of `kubectl` for Kubernetes), not FATE's actual federated-learning engine
-- real FATE requires its own server stack (fate-flow scheduler,
fate-board, a compute engine like Spark/Eggroll), normally deployed via
Docker/Kubernetes. There's no self-contained "run a client and a server
process locally" story here the way there is for
fl_frameworks/flower_adapter.py or fl_frameworks/fedscale_adapter.py --
this would mean standing up unrelated cluster infrastructure this project
doesn't have, not writing an adapter.

Separately, also hit a genuine bug trying to import it further: `fate_
client.pipeline` fails immediately with `AttributeError: "safe_load()"
has been removed, use YAML(typ='safe', pure=True)...` -- fate_client's
own code calls the classic ruamel.yaml top-level API, but the
ruamel.yaml version already installed in this environment (pulled in
earlier by speechbrain's hyperpyyaml dependency) has removed that
function in favor of a new object-oriented interface. A real version
conflict between two unrelated packages' transitive dependencies, not
something specific to this adapter -- noted for completeness, but moot
given the architectural point above already rules this out.
"""
from core.registry import FL_FRAMEWORKS


def build(**kwargs):
    raise NotImplementedError(
        "fate-client is a remote job-submission SDK for an already-deployed FATE "
        "cluster, not a self-contained FL engine -- see this module's docstring."
    )


FL_FRAMEWORKS.register("FATE", implemented=False, organization="WeBank")(build)
