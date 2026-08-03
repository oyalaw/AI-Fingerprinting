import socket
import struct

from core.registry import TRANSPORTS
from transport.base import Transport

_LEN_HEADER = struct.Struct(">I")


def send_framed(sock, data):
    sock.sendall(_LEN_HEADER.pack(len(data)))
    sock.sendall(data)


def recv_exact(sock, n):
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Connection closed while reading")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_framed(sock):
    header = recv_exact(sock, _LEN_HEADER.size)
    (length,) = _LEN_HEADER.unpack(header)
    return recv_exact(sock, length)


class TCPTransport(Transport):
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self._sock = None
        self._server_sock = None

    def connect(self):
        self._sock = socket.create_connection((self.host, self.port))
        return self

    def listen(self):
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))
        self._server_sock.listen(1)
        self._sock, _addr = self._server_sock.accept()
        return self

    def send(self, data):
        send_framed(self._sock, data)

    def recv(self):
        return recv_framed(self._sock)

    def close(self):
        if self._sock:
            self._sock.close()
            self._sock = None
        if self._server_sock:
            self._server_sock.close()
            self._server_sock = None


@TRANSPORTS.register("TCP", implemented=True)
def build_tcp_transport(host, port, **kwargs):
    return TCPTransport(host, port)
