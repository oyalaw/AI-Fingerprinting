import abc


class DistributedFrameworkAdapter(abc.ABC):
    """Interface every entry in the DISTRIBUTED_FRAMEWORKS registry builds an instance of."""

    @abc.abstractmethod
    def run_coordinator(self, config, logger, event_log):
        """Rank-0 process: initializes the process group and coordinates training."""

    @abc.abstractmethod
    def run_worker(self, config, logger, event_log):
        """Rank>0 process: joins the process group and trains alongside the coordinator."""
