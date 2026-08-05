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

        transport = TRANSPORTS.get(self.config.transport).build(
            self.config.host,
            self.config.port,
            tls_cert=self.config.tls_cert,
            tls_key=self.config.tls_key,
        ).listen()

        self.event_log.event("server_listening", host=self.config.host, port=self.config.port)
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
                served += 1
                self.event_log.event("inference_served", request_index=served)
        except ConnectionError:
            self.logger.info("Client disconnected.")
        finally:
            transport.close()
        self.logger.info(f"Server served {served} requests.")

    def _run_federated_learning(self):
        adapter = FL_FRAMEWORKS.get(self.config.fl_framework).build()
        adapter.run_server(self.config, self.logger, self.event_log)

    def _run_distributed_training(self):
        adapter = DISTRIBUTED_FRAMEWORKS.get(self.config.distributed_framework).build()
        adapter.run_coordinator(self.config, self.logger, self.event_log)
