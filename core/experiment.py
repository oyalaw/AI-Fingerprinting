import pathlib
import time

from core.labels import build_ground_truth, new_experiment_id
from core.logger import get_logger
from telemetry.experiment_log import ExperimentLog
from telemetry.ground_truth import write_ground_truth
from telemetry.resource_monitor import ResourceMonitor
from traffic.burst_features import export_bursts
from traffic.flow_features import export_flow_features
from traffic.handcrafted_features import export_features
from traffic.packet_features import ScapyCapture
from traffic.sequence_export import export_sequence


class Experiment:
    """Orchestrator: resolves registry entries, runs the role, captures the
    resulting traffic, and writes ground truth + logs for the run."""

    def __init__(self, config):
        self.config = config
        self.experiment_id = new_experiment_id()
        self.results_dir = pathlib.Path(config.results_dir) / self.experiment_id
        self.results_dir.mkdir(parents=True, exist_ok=True)
        # Uses the full experiment_id, not a truncated slice -- get_logger()
        # keys off this name via logging.getLogger()'s process-wide
        # singleton, so a truncated prefix that collides (e.g. two
        # same-day experiments sharing just a date-only prefix) would
        # silently reuse the first experiment's already-attached file
        # handler instead of logging to this one's own results_dir.
        self.logger = get_logger(f"experiment.{self.experiment_id}", self.results_dir)
        self.event_log = ExperimentLog(self.results_dir / "events.jsonl")

    def run(self):
        timing = {"start": time.time()}
        pcap_path = self.results_dir / f"{self.experiment_id}.pcap"
        capture = None

        if self.config.capture:
            capture = ScapyCapture(
                host=self.config.host,
                port=self.config.port,
                interface=self.config.capture_interface,
                output_path=pcap_path,
                logger=self.logger,
            )
            capture.start()

        resource_path = self.results_dir / f"{self.experiment_id}_{self.config.role}_resource.csv"
        resource_monitor = None
        if self.config.resource_telemetry:
            resource_monitor = ResourceMonitor(
                experiment_id=self.experiment_id,
                role=self.config.role,
                device=self.config.device,
                output_csv=resource_path,
                sample_interval_ms=self.config.resource_sample_interval_ms,
            )
            resource_monitor.start()

        try:
            role = self._build_role()
            role.run()
        finally:
            if capture:
                capture.stop()
            if resource_monitor:
                resource_monitor.stop()

        timing["end"] = time.time()

        sequence_path = flow_features_path = bursts_path = features_path = None
        if capture and pcap_path.exists():
            sequence_path = self.results_dir / f"{self.experiment_id}_sequence.csv"
            export_sequence(pcap_path, sequence_path, server_port=self.config.port)

            flow_features_path = self.results_dir / f"{self.experiment_id}_flow_features.json"
            export_flow_features(pcap_path, flow_features_path, server_port=self.config.port)

            bursts_path = self.results_dir / f"{self.experiment_id}_bursts.csv"
            export_bursts(pcap_path, bursts_path)

            features_path = self.results_dir / f"{self.experiment_id}_features.csv"
            export_features(pcap_path, features_path, server_port=self.config.port, experiment_id=self.experiment_id)

        artifacts = {
            "pcap": str(pcap_path) if capture else None,
            "sequence_csv": str(sequence_path) if sequence_path else None,
            "flow_features": str(flow_features_path) if flow_features_path else None,
            "bursts_csv": str(bursts_path) if bursts_path else None,
            "features_csv": str(features_path) if features_path else None,
            "resource_csv": str(resource_path) if resource_monitor else None,
            "events_log": str(self.results_dir / "events.jsonl"),
        }
        ground_truth = build_ground_truth(self.config, self.experiment_id, timing, artifacts)
        write_ground_truth(self.results_dir / "ground_truth.json", ground_truth)
        self.logger.info(f"Experiment {self.experiment_id} complete. Results in {self.results_dir}")
        return ground_truth

    def _build_role(self):
        from roles.client import ClientRole
        from roles.server import ServerRole
        from roles.standalone import StandaloneRole

        role_map = {"client": ClientRole, "server": ServerRole, "standalone": StandaloneRole}
        role_cls = role_map[self.config.role]
        return role_cls(self.config, self.logger, self.event_log)
