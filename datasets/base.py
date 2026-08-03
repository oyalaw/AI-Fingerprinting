import abc


class Dataset(abc.ABC):
    """Interface every entry in the DATASETS registry builds an instance of."""

    @abc.abstractmethod
    def samples(self, n):
        """Yield up to n (raw_sample, true_label) pairs."""
