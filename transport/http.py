from core.registry import TRANSPORTS


def build_http_transport(host, port, **kwargs):
    raise NotImplementedError("http transport is not yet implemented.")


TRANSPORTS.register("HTTP", implemented=False)(build_http_transport)
