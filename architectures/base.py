"""Convention for entries in the ARCHITECTURES registry.

Each entry's factory has the signature:

    def build(framework_adapter, config) -> model

and is registered with `family=` and `framework=` metadata so
core/config.py can validate that the chosen architecture actually belongs
to the chosen family/framework combo before an experiment starts.

`config` is the full ExperimentConfig -- most architectures ignore it and
just build their one PyTorch module, but it's what lets an architecture
dispatch to a real sub-framework adapter (graph_framework, diffusion_framework,
etc.) when one exists: see architectures/gcn.py and architectures/ddpm.py
for the two real dispatch implementations.
"""


class ArchitectureBuildError(RuntimeError):
    pass
