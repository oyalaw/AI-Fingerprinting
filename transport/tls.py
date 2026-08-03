import datetime
import pathlib
import socket
import ssl

from core.registry import TRANSPORTS
from transport.base import Transport
from transport.tcp import recv_framed, send_framed

_DEV_CERT_DIR = pathlib.Path("experiments/results/.dev_certs")


def ensure_dev_cert(cert_path=None, key_path=None):
    """Return (cert_path, key_path), generating a throwaway self-signed dev
    cert via the `cryptography` package if none is configured, so TLS works
    out of the box for local experiments."""
    if cert_path and key_path:
        return cert_path, key_path

    _DEV_CERT_DIR.mkdir(parents=True, exist_ok=True)
    cert_path = _DEV_CERT_DIR / "dev_cert.pem"
    key_path = _DEV_CERT_DIR / "dev_key.pem"
    if cert_path.exists() and key_path.exists():
        return str(cert_path), str(key_path)

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ai-fingerprint-dev")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
        .sign(key, hashes.SHA256())
    )

    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return str(cert_path), str(key_path)


class TLSTransport(Transport):
    def __init__(self, host, port, cert_path=None, key_path=None):
        self.host = host
        self.port = port
        self.cert_path = cert_path
        self.key_path = key_path
        self._sock = None
        self._server_sock = None

    def connect(self):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE  # research testbed: self-signed dev cert
        raw = socket.create_connection((self.host, self.port))
        self._sock = context.wrap_socket(raw, server_hostname=self.host)
        return self

    def listen(self):
        cert_path, key_path = ensure_dev_cert(self.cert_path, self.key_path)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert_path, key_path)

        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))
        self._server_sock.listen(1)
        raw, _addr = self._server_sock.accept()
        self._sock = context.wrap_socket(raw, server_side=True)
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


@TRANSPORTS.register("TLS", implemented=True)
def build_tls_transport(host, port, tls_cert=None, tls_key=None, **kwargs):
    return TLSTransport(host, port, cert_path=tls_cert, key_path=tls_key)
