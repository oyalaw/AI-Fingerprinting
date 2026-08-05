"""HTTP transport: a plain `http.server`/`http.client` implementation of
the same Transport interface as tcp.py/tls.py -- standard library only,
no new dependency.

HTTP is fundamentally request/response, not the persistent bidirectional
stream TCP/TLS use, so this adapts the project's synchronous send()/recv()
interface onto it with a small queue-based bridge on the server side: the
HTTP handler (its own thread per request, via `ThreadingHTTPServer`) puts
each request body onto a queue for the main thread's recv() to pick up,
then blocks on a second queue waiting for send() to hand it the response
bytes to write back. Only one request is ever in flight at a time here --
correct because this project's client sends one request and waits for the
full response before sending the next (roles/client.py), so the queues
never hold more than one item.

Client side is simpler: connect() just remembers host/port; send() POSTs
via `http.client.HTTPConnection` and stashes the response body; recv()
returns what was stashed.
"""
import http.client
import http.server
import queue
import threading

from core.registry import TRANSPORTS
from transport.base import Transport

_PATH = "/inference"


class HTTPTransport(Transport):
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self._connection = None
        self._pending_response = None
        self._server = None
        self._server_thread = None
        self._request_queue = None
        self._response_queue = None

    def connect(self):
        self._connection = http.client.HTTPConnection(self.host, self.port)
        return self

    def listen(self):
        self._request_queue = queue.Queue()
        self._response_queue = queue.Queue()
        request_queue = self._request_queue
        response_queue = self._response_queue

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # keep stdout clean; this project logs via core/logger.py instead

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                request_queue.put(body)
                response_body = response_queue.get()
                self.send_response(200)
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)

        self._server = http.server.ThreadingHTTPServer((self.host, self.port), Handler)
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        return self

    def send(self, data):
        if self._connection is not None:
            self._connection.request("POST", _PATH, body=data)
            response = self._connection.getresponse()
            self._pending_response = response.read()
        else:
            self._response_queue.put(data)

    def recv(self):
        if self._connection is not None:
            return self._pending_response
        return self._request_queue.get()

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


@TRANSPORTS.register("HTTP", implemented=True)
def build_http_transport(host, port, **kwargs):
    return HTTPTransport(host, port)
