from core.registry import APPLICATIONS, ARCHITECTURES, DATASETS, FRAMEWORKS
from roles.base import Role


class StandaloneRole(Role):
    """Runs the inference workload in-process with no network/capture — for
    fast dev/debug of a new framework/architecture/application/dataset combo
    before wiring it up to a real client/server pair."""

    def run(self):
        framework = FRAMEWORKS.get(self.config.framework).build()
        architecture_entry = ARCHITECTURES.get(self.config.architecture)
        model = framework.load_model(architecture_entry)
        application = APPLICATIONS.get(self.config.application).build()
        dataset = DATASETS.get(self.config.dataset).build()

        for i, (raw_sample, true_label) in enumerate(dataset.samples(self.config.num_requests)):
            input_tensor = application.preprocess(raw_sample)
            output_tensor = framework.predict(model, input_tensor)
            result = application.postprocess(output_tensor)
            self.logger.info(f"[standalone {i}] predicted={result} true={true_label}")
            self.event_log.event(
                "standalone_inference", index=i, predicted=result, true_label=true_label
            )
