from core.registry import (
    ARCHITECTURES,
    DISTRIBUTED_FRAMEWORKS,
    FL_FRAMEWORKS,
    FRAMEWORKS,
    TRANSPORTS,
)
from roles.base import Role


class ServerRole(Role):
    """Model-hosting / aggregating side. For paradigm=inference this loads a
    real model and serves requests. For federated_learning / distributed_
    training, it delegates to the selected FL/distributed adapter, which
    manages its own server protocol (e.g. Flower's own gRPC aggregator)."""

    def run(self):
        if self.config.paradigm == "federated_learning":
            return self._run_federated_learning()
        if self.config.paradigm == "distributed_training":
            return self._run_distributed_training()
        return self._run_inference()

    def _run_inference(self):
        framework = FRAMEWORKS.get(self.config.framework).build()
        architecture_entry = ARCHITECTURES.get(self.config.architecture)
        model = framework.load_model(architecture_entry, self.config)

        self.event_log.event(
            "server_start", listen_host=self.config.host, listen_port=self.config.port, transport=self.config.transport
        )
        transport = TRANSPORTS.get(self.config.transport).build(
            self.config.host,
            self.config.port,
            tls_cert=self.config.tls_cert,
            tls_key=self.config.tls_key,
        ).listen()
        self.event_log.event("client_connected", peer=transport.peer_address)

        self.logger.info(
            f"Server listening on {self.config.host}:{self.config.port} "
            f"({self.config.framework}/{self.config.architecture}/{self.config.application})"
        )

        served = 0
        try:
            while served < self.config.num_requests:
                request = transport.recv()
                input_tensor = framework.deserialize(request)
                output_tensor = framework.predict(model, input_tensor)
                response = framework.serialize(output_tensor)
                transport.send(response)
                # req-N naming (0-indexed by the count *before*
                # incrementing) matches roles/client.py's own request_id
                # exactly -- correct here specifically because this
                # protocol is synchronous single-client request/response,
                # so the Nth request received server-side is guaranteed
                # to be the Nth request sent client-side. The wire
                # protocol itself carries no request-id field, so this
                # positional correlation is the only link between the two
                # sides' event logs.
                self.event_log.event(
                    "inference_completed",
                    request_id=f"req-{served}",
                    input_bytes=len(request),
                    output_bytes=len(response),
                )
                served += 1
        except ConnectionError:
            self.event_log.event("client_disconnected", peer=transport.peer_address)
            self.logger.info("Client disconnected.")
        except Exception as exc:
            self.event_log.event(
                "server_error", peer=transport.peer_address, error_type=type(exc).__name__, error=str(exc)
            )
            raise
        finally:
            transport.close()
            self.event_log.event("server_stop")
        self.logger.info(f"Server served {served} requests.")

    def _run_federated_learning(self):
        adapter = FL_FRAMEWORKS.get(self.config.fl_framework).build()
        adapter.run_server(self.config, self.logger, self.event_log)

    def _run_distributed_training(self):
        adapter = DISTRIBUTED_FRAMEWORKS.get(self.config.distributed_framework).build()
        adapter.run_coordinator(self.config, self.logger, self.event_log)
