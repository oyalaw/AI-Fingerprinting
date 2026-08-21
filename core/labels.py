"""Ground-truth label schema: the 6-level fingerprinting taxonomy this whole
pipeline exists to produce labeled encrypted-traffic traces for.

    Level 1  Framework       e.g. PyTorch, TensorRT, Flower
    Level 2  Family          e.g. CNN, Transformer
    Level 3  Architecture    e.g. ResNet18, BERT
    Level 4  Application     e.g. Image Classification
    Level 5  Device          e.g. Jetson AGX Orin, iPhone (core/devices.py)
    Level 6  OS              e.g. JetPack, Windows (core/operating_systems.py)

Device and OS are deliberately separate axes (Level 5 and Level 6), not one
combined "device" -- see core/devices.py's own docstring for why: a single
device (a DGX) can plausibly run more than one OS, so folding them into one
field (as this project originally did, treating "ubuntu"/"windows" as if
they were devices themselves) breaks down.
"""
import datetime
import uuid


def new_experiment_id():
    # A pure timestamp risks two experiments started in the same second
    # colliding and silently overwriting each other's results_dir -- keeping
    # a short uuid suffix guarantees uniqueness while still sorting/reading
    # chronologically, since this value names both the results folder and
    # every artifact filename inside it (see core/experiment.py).
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"


def build_ground_truth(config, experiment_id, timing, artifacts):
    return {
        "experiment_id": experiment_id,
        "timestamp_start": timing.get("start"),
        "timestamp_end": timing.get("end"),
        "paradigm": config.paradigm,
        "role": config.role,
        "labels": {
            "level1_framework": config.framework,
            "level2_family": config.family,
            "level3_architecture": config.architecture,
            "level4_application": config.application,
            "level5_device": config.device,
            "level6_os": config.operating_system,
        },
        "sub_frameworks": {
            "fl_framework": config.fl_framework,
            "distributed_framework": config.distributed_framework,
            "llm_framework": config.llm_framework,
            "cv_framework": config.cv_framework,
            "speech_framework": config.speech_framework,
            "graph_framework": config.graph_framework,
            "diffusion_framework": config.diffusion_framework,
        },
        "dataset": config.dataset,
        "transport": config.transport,
        "network": {"host": config.host, "port": config.port},
        "num_requests": config.num_requests,
        "batch_size": config.batch_size,
        "artifacts": artifacts,
    }
