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

    @property
    def peer_address(self):
        """"host:port" of the connected remote endpoint, or None when a
        transport doesn't expose one (default here) -- only
        transport/tcp.py overrides this today; the others (HTTP/gRPC/TLS)
        are a real, documented gap, not silently pretended to work."""
        return None
