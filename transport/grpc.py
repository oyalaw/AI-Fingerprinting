"""gRPC transport — stub.

Note: this is deliberately NOT needed to make the Flower federated-learning
slice work. Flower brings its own gRPC client/server implementation
internally (`flwr.server.start_server` / `flwr.client.start_client`) — our
role classes just start Flower with the right config and let scapy capture
whatever port it opens. This module is for anything else that wants a plain
generic gRPC transport matching the Transport interface; building that
correctly (framing arbitrary bytes over a fundamentally request/response
unary-RPC model, without a protobuf schema) is real work, so it's left as a
stub rather than shipping a shaky hack.
"""
from core.registry import TRANSPORTS


def build_grpc_transport(host, port, **kwargs):
    raise NotImplementedError("grpc transport is not yet implemented; see module docstring.")


TRANSPORTS.register("gRPC", implemented=False)(build_grpc_transport)
