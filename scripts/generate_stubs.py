"""One-off code generator: writes every stub registration file for the
project. Run with `python scripts/generate_stubs.py` from the repo root.
Re-running is safe/idempotent -- it always overwrites with the same content.

This exists because the ~90 remaining registry entries (everything not part
of the one fully-implemented vertical slice per registry) are mechanically
identical: register the name, raise NotImplementedError if anyone tries to
build it. Writing them by hand one at a time would be pure repetition.
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

ADAPTER_TEMPLATE = '''"""{display_name} -- registered but not yet implemented.{extra_doc}"""
from core.registry import {registry_var}


def build(**kwargs):
    raise NotImplementedError("{display_name} is not yet implemented.")


{registry_var}.register("{display_name}", implemented=False{meta_kwargs})(build)
'''

FAMILY_TEMPLATE = '''"""{display_name} family -- registered but not yet implemented."""
from core.registry import FAMILIES

FAMILIES.add("{display_name}", implemented=False, description="{description}")
'''

ARCHITECTURE_TEMPLATE = '''"""{display_name} -- registered but not yet implemented."""
from core.registry import ARCHITECTURES


def build(framework_adapter):
    raise NotImplementedError("{display_name} is not yet implemented.")


ARCHITECTURES.register(
    "{display_name}", implemented=False, family="{family}", framework="{framework}"
)(build)
'''

APPLICATION_TEMPLATE = '''"""{display_name} application -- registered but not yet implemented."""
from applications.base import Application
from core.registry import APPLICATIONS


class {class_name}(Application):
    def preprocess(self, raw_sample):
        raise NotImplementedError("{display_name} is not yet implemented.")

    def postprocess(self, output_tensor):
        raise NotImplementedError("{display_name} is not yet implemented.")


APPLICATIONS.register("{display_name}", implemented=False)({class_name})
'''

DATASET_TEMPLATE = '''"""{display_name} dataset -- registered but not yet implemented."""
from core.registry import DATASETS
from datasets.base import Dataset


class {class_name}(Dataset):
    def samples(self, n):
        raise NotImplementedError("{display_name} is not yet implemented.")


DATASETS.register("{display_name}", implemented=False)({class_name})
'''


def meta_kwargs_str(meta):
    if not meta:
        return ""
    parts = []
    for key, value in meta.items():
        parts.append(f", {key}={value!r}")
    return "".join(parts)


def slugify(name):
    return (
        name.lower()
        .replace(" ", "_")
        .replace(".", "")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )


def class_name_for(name):
    return "".join(part.capitalize() for part in slugify(name).split("_")) or "Entry"


def write_adapter_stub(directory, registry_var, display_name, meta, extra_doc="", filename=None):
    directory = ROOT / directory
    directory.mkdir(parents=True, exist_ok=True)
    filename = filename or f"{slugify(display_name)}_adapter.py"
    content = ADAPTER_TEMPLATE.format(
        display_name=display_name,
        registry_var=registry_var,
        meta_kwargs=meta_kwargs_str(meta),
        extra_doc=(" " + extra_doc) if extra_doc else "",
    )
    (directory / filename).write_text(content, encoding="utf-8")


def write_family_stub(display_name, description):
    directory = ROOT / "families" / slugify(display_name)
    directory.mkdir(parents=True, exist_ok=True)
    content = FAMILY_TEMPLATE.format(display_name=display_name, description=description)
    (directory / "__init__.py").write_text(content, encoding="utf-8")


def write_architecture_stub(display_name, family, framework, filename=None):
    directory = ROOT / "architectures"
    filename = filename or f"{slugify(display_name)}.py"
    content = ARCHITECTURE_TEMPLATE.format(display_name=display_name, family=family, framework=framework)
    (directory / filename).write_text(content, encoding="utf-8")


def write_application_stub(display_name, filename=None):
    directory = ROOT / "applications"
    filename = filename or f"{slugify(display_name)}.py"
    content = APPLICATION_TEMPLATE.format(display_name=display_name, class_name=class_name_for(display_name))
    (directory / filename).write_text(content, encoding="utf-8")


def write_dataset_stub(display_name, filename=None):
    directory = ROOT / "datasets"
    filename = filename or f"{slugify(display_name)}.py"
    content = DATASET_TEMPLATE.format(display_name=display_name, class_name=class_name_for(display_name))
    (directory / filename).write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Data: every remaining stub entry, taken directly from the user's tables.
# ---------------------------------------------------------------------------

FRAMEWORKS = [
    ("TensorFlow", {"organization": "Google", "platforms": ["windows", "linux", "macos", "jetson"]}),
    ("ONNX Runtime", {"organization": "Microsoft", "platforms": ["windows", "linux", "macos", "jetson"]}),
    ("ONNX Runtime Mobile", {"organization": "Microsoft", "platforms": ["android", "ios", "ipados"]}),
    ("TensorRT", {"organization": "NVIDIA", "platforms": ["jetson"]}),
    ("DeepStream", {"organization": "NVIDIA", "platforms": ["jetson"]}),
    ("OpenVINO", {"organization": "Intel", "platforms": ["windows", "linux", "macos"]}),
    ("TVM", {"organization": "Apache", "platforms": ["windows", "linux", "macos", "android", "jetson"]}),
    ("Edge Impulse", {"organization": "Edge Impulse", "platforms": ["linux", "jetson"]}),
    ("TensorFlow Lite", {"organization": "Google", "platforms": ["android", "ios", "ipados", "linux", "windows", "macos", "jetson"]}),
    ("TensorFlow Lite Micro", {"organization": "Google", "platforms": ["android"]}),
    ("PyTorch Mobile", {"organization": "Meta", "platforms": ["android", "ios", "ipados"]}),
    ("ExecuTorch", {"organization": "Meta", "platforms": ["android", "ios", "ipados", "macos"]}),
    ("CoreML", {"organization": "Apple", "platforms": ["ios", "ipados", "macos"]}),
    ("NNAPI", {"organization": "Google", "platforms": ["android"]}),
    ("Arm NN", {"organization": "Arm", "platforms": ["android", "linux", "jetson"]}),
    ("MNN", {"organization": "Alibaba", "platforms": ["android", "ios", "ipados"]}),
    ("NCNN", {"organization": "Tencent", "platforms": ["android", "ios", "ipados", "linux"]}),
    ("RKNN", {"organization": "Rockchip", "platforms": ["linux"]}),
    ("SNPE", {"organization": "Qualcomm", "platforms": ["android"]}),
    ("MediaPipe", {"organization": "Google", "platforms": ["android", "ios", "ipados", "macos"]}),
    ("MPS Backend", {"organization": "Apple", "platforms": ["macos"]}),
    ("Metal Performance Shaders", {"organization": "Apple", "platforms": ["ios", "ipados", "macos"]}),
]

FL_FRAMEWORKS = [
    ("TensorFlow Federated", {"organization": "Google"}),
    ("FedML", {"organization": "FedML Inc."}),
    ("PySyft", {"organization": "OpenMined"}),
    ("FATE", {"organization": "WeBank"}),
    ("OpenFL", {"organization": "Intel"}),
    ("NVFlare", {"organization": "NVIDIA"}),
    ("PaddleFL", {"organization": "Baidu"}),
    ("Clara FL", {"organization": "NVIDIA"}),
    ("FedScale", {"organization": "Michigan State University"}),
    ("LEAF", {"organization": "Carnegie Mellon University"}),
    ("EasyFL", {"organization": "Microsoft"}),
    ("FedJAX", {"organization": "Google"}),
    ("FedGraph", {"organization": "Rutgers University"}),
    ("APPFL", {"organization": "Argonne National Laboratory"}),
    ("FedLab", {"organization": "SMILELab"}),
]

DISTRIBUTED_FRAMEWORKS = [
    ("Horovod", {"organization": "Uber"}),
    ("DeepSpeed", {"organization": "Microsoft"}),
    ("ColossalAI", {"organization": "HPC AI Tech"}),
    ("Megatron LM", {"organization": "NVIDIA"}),
    ("Ray Train", {"organization": "Anyscale"}),
    ("BytePS", {"organization": "Bytedance"}),
    ("Alpa", {"organization": "Stanford"}),
    ("FairScale", {"organization": "Meta"}),
    ("ZeRO", {"organization": "Microsoft"}),
]

LLM_FRAMEWORKS = [
    ("HuggingFace Transformers", {"use_case": "NLP"}),
    ("vLLM", {"use_case": "High throughput inference"}),
    ("Ollama", {"use_case": "Local LLM serving"}),
    ("LMDeploy", {"use_case": "Efficient deployment"}),
    ("TensorRT LLM", {"use_case": "NVIDIA inference"}),
    ("SGLang", {"use_case": "Structured generation"}),
    ("llama.cpp", {"use_case": "CPU inference"}),
    ("ExLlama", {"use_case": "Quantized LLMs"}),
    ("FastChat", {"use_case": "LLM serving"}),
    ("Text Generation Inference", {"use_case": "HuggingFace"}),
]

CV_FRAMEWORKS = [
    ("Ultralytics", {"models": "YOLOv5, YOLOv8, YOLOv11"}),
    ("MMDetection", {"models": "Detection models"}),
    ("Detectron2", {"models": "Meta vision models"}),
    ("OpenMMLab", {"models": "Vision toolkit"}),
    ("YOLOX", {"models": "Detection"}),
    ("PaddleDetection", {"models": "Detection"}),
    ("Segment Anything", {"models": "Segmentation"}),
    ("MMSegmentation", {"models": "Segmentation"}),
]

SPEECH_FRAMEWORKS = [
    ("ESPnet", {"application": "Speech recognition"}),
    ("Kaldi", {"application": "ASR"}),
    ("SpeechBrain", {"application": "Speech tasks"}),
    ("NeMo", {"application": "NVIDIA speech"}),
    ("Coqui STT", {"application": "Speech to text"}),
    ("Whisper", {"application": "Speech recognition"}),
]

GRAPH_FRAMEWORKS = [
    ("PyTorch Geometric", {"application": "GNN"}),
    ("DGL", {"application": "GNN"}),
    ("StellarGraph", {"application": "GNN"}),
    ("Spektral", {"application": "GNN"}),
]

DIFFUSION_FRAMEWORKS = [
    ("Diffusers", {"application": "HuggingFace"}),
    ("Stable Diffusion WebUI", {"application": "Image generation"}),
    ("ComfyUI", {"application": "Workflow based generation"}),
    ("InvokeAI", {"application": "Image generation"}),
]

FAMILIES = [
    ("RNN", "Recurrent Neural Networks"),
    ("Transformer", "Transformer architectures"),
    ("GNN", "Graph Neural Networks"),
    ("Autoencoder", "Autoencoders"),
    ("Diffusion", "Diffusion models"),
]

ARCHITECTURES = [
    ("ResNet50", "CNN", "PyTorch"),
    ("MobileNetV2", "CNN", "PyTorch"),
    ("YOLOv8", "CNN", "PyTorch"),
    ("LSTM", "RNN", "PyTorch"),
    ("GRU", "RNN", "PyTorch"),
    ("BERT", "Transformer", "PyTorch"),
    ("DistilBERT", "Transformer", "PyTorch"),
    ("ViT", "Transformer", "PyTorch"),
    ("GCN", "GNN", "PyTorch"),
    ("DDPM", "Diffusion", "PyTorch"),
]

APPLICATIONS = [
    "Object Detection",
    "Sentiment Analysis",
    "Activity Recognition",
    "Text Generation",
    "Image Generation",
    "Segmentation",
    "Speech Recognition",
]

DATASETS = [
    "COCO",
    "ImageNet",
    "UCI HAR",
    "IMDB",
    "SST2",
]


def main():
    for name, meta in FRAMEWORKS:
        write_adapter_stub("frameworks", "FRAMEWORKS", name, meta)
    for name, meta in FL_FRAMEWORKS:
        write_adapter_stub("fl_frameworks", "FL_FRAMEWORKS", name, meta)
    for name, meta in DISTRIBUTED_FRAMEWORKS:
        write_adapter_stub("distributed_frameworks", "DISTRIBUTED_FRAMEWORKS", name, meta)
    for name, meta in LLM_FRAMEWORKS:
        write_adapter_stub("llm_frameworks", "LLM_FRAMEWORKS", name, meta)
    for name, meta in CV_FRAMEWORKS:
        write_adapter_stub("cv_frameworks", "CV_FRAMEWORKS", name, meta)
    for name, meta in SPEECH_FRAMEWORKS:
        write_adapter_stub("speech_frameworks", "SPEECH_FRAMEWORKS", name, meta)
    for name, meta in GRAPH_FRAMEWORKS:
        write_adapter_stub("graph_frameworks", "GRAPH_FRAMEWORKS", name, meta)
    for name, meta in DIFFUSION_FRAMEWORKS:
        write_adapter_stub("diffusion_frameworks", "DIFFUSION_FRAMEWORKS", name, meta)

    for name, description in FAMILIES:
        write_family_stub(name, description)

    for name, family, framework in ARCHITECTURES:
        write_architecture_stub(name, family, framework)

    for name in APPLICATIONS:
        write_application_stub(name)

    for name in DATASETS:
        write_dataset_stub(name)

    total = (
        len(FRAMEWORKS) + len(FL_FRAMEWORKS) + len(DISTRIBUTED_FRAMEWORKS) + len(LLM_FRAMEWORKS)
        + len(CV_FRAMEWORKS) + len(SPEECH_FRAMEWORKS) + len(GRAPH_FRAMEWORKS) + len(DIFFUSION_FRAMEWORKS)
        + len(FAMILIES) + len(ARCHITECTURES) + len(APPLICATIONS) + len(DATASETS)
    )
    print(f"Generated {total} stub files.")


if __name__ == "__main__":
    main()
