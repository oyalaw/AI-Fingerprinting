"""Generic plugin registry used by every pluggable axis in the project
(frameworks, families, architectures, applications, datasets, transports,
roles, devices, and every framework-ecosystem registry).

A stub entry (implemented=False) still registers and shows up in --list /
--interactive, but raises NotImplementedError if something tries to build it.
That's the whole point: the full intended matrix is always visible, even
before every cell has a real implementation behind it.
"""
import importlib
import pkgutil


class RegistryEntry:
    def __init__(self, name, factory, implemented, **meta):
        self.name = name
        self.factory = factory
        self.implemented = implemented
        self.meta = meta

    def build(self, *args, **kwargs):
        if not self.implemented:
            raise NotImplementedError(
                f"'{self.name}' is registered but not yet implemented. "
                f"Implement {self.factory.__module__}.{self.factory.__name__} "
                f"and set implemented=True to enable it."
            )
        return self.factory(*args, **kwargs)

    def __repr__(self):
        status = "implemented" if self.implemented else "stub"
        return f"<{self.name} ({status})>"


class Registry:
    def __init__(self, category):
        self.category = category
        self._entries = {}

    def register(self, name, implemented=True, **meta):
        def decorator(factory):
            key = name.lower()
            if key in self._entries:
                raise ValueError(f"{self.category} '{name}' is already registered")
            self._entries[key] = RegistryEntry(name, factory, implemented, **meta)
            return factory
        return decorator

    def add(self, name, factory=None, implemented=True, **meta):
        """Non-decorator form, for registries with no meaningful factory (e.g. devices)."""
        self.register(name, implemented=implemented, **meta)(factory or (lambda: None))

    def has(self, name):
        return bool(name) and name.lower() in self._entries

    def get(self, name):
        key = (name or "").lower()
        if key not in self._entries:
            available = ", ".join(sorted(e.name for e in self._entries.values()))
            raise KeyError(f"Unknown {self.category} '{name}'. Available: {available}")
        return self._entries[key]

    def list(self, implemented_only=False):
        entries = sorted(self._entries.values(), key=lambda e: e.name.lower())
        if implemented_only:
            entries = [e for e in entries if e.implemented]
        return entries

    def __len__(self):
        return len(self._entries)

    def __iter__(self):
        return iter(self.list())


FRAMEWORKS = Registry("framework")
FL_FRAMEWORKS = Registry("fl_framework")
DISTRIBUTED_FRAMEWORKS = Registry("distributed_framework")
LLM_FRAMEWORKS = Registry("llm_framework")
CV_FRAMEWORKS = Registry("cv_framework")
SPEECH_FRAMEWORKS = Registry("speech_framework")
GRAPH_FRAMEWORKS = Registry("graph_framework")
DIFFUSION_FRAMEWORKS = Registry("diffusion_framework")
FAMILIES = Registry("family")
ARCHITECTURES = Registry("architecture")
APPLICATIONS = Registry("application")
DATASETS = Registry("dataset")
TRANSPORTS = Registry("transport")
DEVICES = Registry("device")
OPERATING_SYSTEMS = Registry("operating_system")

ALL_REGISTRIES = {
    "framework": FRAMEWORKS,
    "fl_framework": FL_FRAMEWORKS,
    "distributed_framework": DISTRIBUTED_FRAMEWORKS,
    "llm_framework": LLM_FRAMEWORKS,
    "cv_framework": CV_FRAMEWORKS,
    "speech_framework": SPEECH_FRAMEWORKS,
    "graph_framework": GRAPH_FRAMEWORKS,
    "diffusion_framework": DIFFUSION_FRAMEWORKS,
    "family": FAMILIES,
    "architecture": ARCHITECTURES,
    "application": APPLICATIONS,
    "dataset": DATASETS,
    "transport": TRANSPORTS,
    "device": DEVICES,
    "operating_system": OPERATING_SYSTEMS,
}

_DISCOVERABLE_PACKAGES = (
    "frameworks",
    "fl_frameworks",
    "distributed_frameworks",
    "llm_frameworks",
    "cv_frameworks",
    "speech_frameworks",
    "graph_frameworks",
    "diffusion_frameworks",
    "families",
    "architectures",
    "applications",
    "datasets",
    "transport",
)

_discovered = False


def discover(package_name):
    """Import every submodule of *package_name* so its @register(...) decorators run."""
    package = importlib.import_module(package_name)
    if not hasattr(package, "__path__"):
        return
    for _, module_name, is_pkg in pkgutil.iter_modules(package.__path__, prefix=f"{package_name}."):
        importlib.import_module(module_name)
        if is_pkg:
            discover(module_name)


def discover_all():
    """Populate every registry above by importing all plugin packages. Idempotent."""
    global _discovered
    if _discovered:
        return
    import core.devices  # noqa: F401 - registers DEVICES entries
    import core.operating_systems  # noqa: F401 - registers OPERATING_SYSTEMS entries
    for package_name in _DISCOVERABLE_PACKAGES:
        discover(package_name)
    _discovered = True
