import abc


class Role(abc.ABC):
    def __init__(self, config, logger, event_log):
        self.config = config
        self.logger = logger
        self.event_log = event_log

    @abc.abstractmethod
    def run(self):
        ...
