from core.registry import (
    APPLICATIONS,
    DATASETS,
    DISTRIBUTED_FRAMEWORKS,
    FL_FRAMEWORKS,
    FRAMEWORKS,
    TRANSPORTS,
)
from roles.base import Role
from telemetry.timestamps import RequestTimer


class ClientRole(Role):
    """Traffic-generating side. For paradigm=inference this is a real
    request/response client. For federated_learning / distributed_training,
    it delegates to the selected FL/distributed adapter, which manages its
    own client protocol (e.g. Flower's own gRPC client)."""

    def run(self):
        if self.config.paradigm == "federated_learning":
            return self._run_federated_learning()
        if self.config.paradigm == "distributed_training":
            return self._run_distributed_training()
        return self._run_inference()

    def _run_inference(self):
        framework = FRAMEWORKS.get(self.config.framework).build()
        application = APPLICATIONS.get(self.config.application).build()
        dataset = DATASETS.get(self.config.dataset).build()

        transport = TRANSPORTS.get(self.config.transport).build(
            self.config.host,
            self.config.port,
            tls_cert=self.config.tls_cert,
            tls_key=self.config.tls_key,
        ).connect()

        timer = RequestTimer()
        self.event_log.event("client_connected", host=self.config.host, port=self.config.port)
        self.logger.info(f"Connected to {self.config.host}:{self.config.port} via {self.config.transport}")

        try:
            for i, (raw_sample, true_label) in enumerate(dataset.samples(self.config.num_requests)):
                request_id = f"req-{i}"
                input_tensor = application.preprocess(raw_sample)
                payload = framework.serialize(input_tensor)

                timer.start(request_id)
                transport.send(payload)
                response = transport.recv()
                record = timer.stop(request_id, bytes_sent=len(payload), bytes_received=len(response))

                output_tensor = framework.deserialize(response)
                result = application.postprocess(output_tensor)
                self.event_log.event(
                    "inference_response",
                    request_id=request_id,
                    true_label=true_label,
                    predicted=result,
                    duration_s=record["duration_s"],
                )
                self.logger.info(
                    f"[{request_id}] predicted={result} true={true_label} "
                    f"({record['duration_s'] * 1000:.1f} ms)"
                )
        finally:
            transport.close()

    def _run_federated_learning(self):
        adapter = FL_FRAMEWORKS.get(self.config.fl_framework).build()
        adapter.run_client(self.config, self.logger, self.event_log)

    def _run_distributed_training(self):
        adapter = DISTRIBUTED_FRAMEWORKS.get(self.config.distributed_framework).build()
        adapter.run_worker(self.config, self.logger, self.event_log)
