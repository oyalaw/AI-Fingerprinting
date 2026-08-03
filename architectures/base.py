"""Convention for entries in the ARCHITECTURES registry.

Each entry's factory has the signature:

    def build(framework_adapter) -> model

and is registered with `family=` and `framework=` metadata so
core/config.py can validate that the chosen architecture actually belongs
to the chosen family/framework combo before an experiment starts.
"""


class ArchitectureBuildError(RuntimeError):
    pass
