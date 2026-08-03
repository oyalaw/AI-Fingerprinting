import abc


class FLFrameworkAdapter(abc.ABC):
    """Interface every entry in the FL_FRAMEWORKS registry builds an instance of."""

    @abc.abstractmethod
    def run_server(self, config, logger, event_log):
        """Start the FL aggregation server. Blocks until num_rounds complete."""

    @abc.abstractmethod
    def run_client(self, config, logger, event_log):
        """Connect to the FL server and participate as one client for the run."""
