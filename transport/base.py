import abc


class Transport(abc.ABC):
    @abc.abstractmethod
    def connect(self):
        """Client-side: establish a connection to (host, port). Returns self."""

    @abc.abstractmethod
    def listen(self):
        """Server-side: bind and accept one connection. Returns self."""

    @abc.abstractmethod
    def send(self, data: bytes):
        ...

    @abc.abstractmethod
    def recv(self) -> bytes:
        ...

    @abc.abstractmethod
    def close(self):
        ...
