import abc


class Application(abc.ABC):
    """Interface every entry in the APPLICATIONS registry builds an instance of."""

    @abc.abstractmethod
    def preprocess(self, raw_sample):
        """raw_sample (from a Dataset) -> input tensor ready for the framework adapter."""

    @abc.abstractmethod
    def postprocess(self, output_tensor):
        """model output tensor -> a human/ML-readable result."""
