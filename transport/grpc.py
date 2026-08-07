"""gRPC transport -- real implementation, same `Transport` interface as
transport/tcp.py, tls.py, http.py.

Note: this is deliberately NOT needed to make the Flower federated-learning
slice work. Flower brings its own gRPC client/server implementation
internally (`flwr.server.start_server` / `flwr.client.start_client`) --
our role classes just start Flower with the right config and let scapy
capture whatever port it opens. This module is a plain generic gRPC
transport for `paradigm: inference` matching the same interface TCP/TLS/
HTTP already provide.

Previously a stub: framing arbitrary bytes over gRPC's fundamentally
request/response unary-RPC model, without a protobuf schema, looked like
it would need one. It doesn't -- gRPC's low-level Python API supports
raw-bytes unary-unary calls directly, with no `.proto` file or generated
stub code, by registering a single RPC method with identity
serializers/deserializers (`request_deserializer=lambda x: x`,
`response_serializer=lambda x: x`) via `grpc.unary_unary_rpc_method_handler`
+ `grpc.method_handlers_generic_handler`. Confirmed directly with a
minimal client/server roundtrip before writing this file: a real gRPC
server receiving and echoing arbitrary bytes over an insecure channel, no
compiled schema anywhere.

gRPC's own unary call is naturally a single request-in/response-out unit,
but this project's `Transport` interface splits that into separate
`send()`/`recv()` calls (to stay symmetric with TCP/TLS's persistent-
socket shape) -- bridged the same way transport/http.py already bridges
HTTP's request/response shape onto that interface: a small queue pair.
Server side: the RPC handler (its own thread per call, via grpc's
`ThreadPoolExecutor`) puts the request body on a queue for the main
thread's `recv()` to pick up, then blocks on a second queue for `send()`
to hand back the response bytes. Client side is simpler: `connect()` just
opens the channel and resolves the raw unary-unary callable; `send()`
performs the actual RPC call and stashes the response; `recv()` returns
what was stashed -- same shape as http.py's client side.

`grpc` is imported lazily inside methods, not at module scope, matching
every other framework/transport adapter here (`python main.py --list`
stays viewable without dependencies installed) -- and note this module's
own name (`transport/grpc.py`) doesn't collide with the real `grpc`
package on import: Python 3's absolute-import default means `import grpc`
from inside `transport.grpc` resolves the top-level site-packages `grpc`,
not this module itself; confirmed directly, unlike the real collision
`datasets/imdb.py`'s docstring documents between this project's own
top-level `datasets/` package and HuggingFace's `datasets` pip package.
"""
import queue
import threading

from core.registry import TRANSPORTS
from transport.base import Transport

_SERVICE_NAME = "FingerprintingTransport"
_METHOD_NAME = "Exchange"
_FULL_METHOD = f"/{_SERVICE_NAME}/{_METHOD_NAME}"


def _identity(data):
    return data


class GRPCTransport(Transport):
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self._channel = None
        self._call = None
        self._pending_response = None
        self._server = None
        self._request_queue = None
        self._response_queue = None

    def connect(self):
        import grpc

        self._channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        self._call = self._channel.unary_unary(
            _FULL_METHOD, request_serializer=_identity, response_deserializer=_identity
        )
        return self

    def listen(self):
        import grpc

        self._request_queue = queue.Queue()
        self._response_queue = queue.Queue()
        request_queue = self._request_queue
        response_queue = self._response_queue

        def handler(request_bytes, context):
            request_queue.put(request_bytes)
            return response_queue.get()

        rpc_method_handler = grpc.unary_unary_rpc_method_handler(
            handler, request_deserializer=_identity, response_serializer=_identity
        )
        generic_handler = grpc.method_handlers_generic_handler(
            _SERVICE_NAME, {_METHOD_NAME: rpc_method_handler}
        )

        from concurrent.futures import ThreadPoolExecutor

        self._server = grpc.server(ThreadPoolExecutor(max_workers=4))
        self._server.add_generic_rpc_handlers((generic_handler,))
        self._server.add_insecure_port(f"{self.host}:{self.port}")
        self._server.start()
        return self

    def send(self, data):
        if self._call is not None:
            self._pending_response = self._call(data)
        else:
            self._response_queue.put(data)

    def recv(self):
        if self._call is not None:
            return self._pending_response
        return self._request_queue.get()

    def close(self):
        if self._channel is not None:
            self._channel.close()
            self._channel = None
        if self._server is not None:
            self._server.stop(grace=None)
            self._server = None


@TRANSPORTS.register("gRPC", implemented=True)
def build_grpc_transport(host, port, **kwargs):
    return GRPCTransport(host, port)
