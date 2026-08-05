import abc


class FrameworkAdapter(abc.ABC):
    """Interface every entry in the FRAMEWORKS registry builds an instance of."""

    @abc.abstractmethod
    def load_model(self, architecture_entry, config):
        """Build and return a ready-to-serve model for the given ARCHITECTURES
        registry entry (server-side only; clients never load a model).

        `config` is the full ExperimentConfig, passed through unchanged so
        architecture build() functions can dispatch on config.graph_framework/
        cv_framework/diffusion_framework/etc. when a real sub-framework
        adapter exists to build an equivalent model -- see architectures/gcn.py
        and architectures/ddpm.py for the two real dispatch implementations."""

    @abc.abstractmethod
    def predict(self, model, input_tensor):
        ...

    @abc.abstractmethod
    def serialize(self, tensor) -> bytes:
        ...

    @abc.abstractmethod
    def deserialize(self, data: bytes):
        ...
