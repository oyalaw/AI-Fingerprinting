import abc


class FrameworkAdapter(abc.ABC):
    """Interface every entry in the FRAMEWORKS registry builds an instance of."""

    @abc.abstractmethod
    def load_model(self, architecture_entry):
        """Build and return a ready-to-serve model for the given ARCHITECTURES
        registry entry (server-side only; clients never load a model)."""

    @abc.abstractmethod
    def predict(self, model, input_tensor):
        ...

    @abc.abstractmethod
    def serialize(self, tensor) -> bytes:
        ...

    @abc.abstractmethod
    def deserialize(self, data: bytes):
        ...
