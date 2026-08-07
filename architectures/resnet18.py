"""ResNet18 -- the one fully-implemented architecture this pass. Random init
(not pretrained) so no network download is required at model-load time; the
point of this project is the traffic it generates, not classification
accuracy. num_classes=10 to match CIFAR10. torchvision is imported lazily
inside build() so registering this architecture doesn't require it
installed.

Natively built with PyTorch, but `also_supports` every other implemented
framework: TensorRT, OpenVINO, TensorFlow Lite, ONNX Runtime, ONNX Runtime
Mobile, PyTorch Mobile, ExecuTorch, TVM, CoreML, TensorFlow, TensorFlow
Lite Micro, MediaPipe, NCNN, SNPE, RKNN, Arm NN, DeepStream, MNN, and
NNAPI all convert/compile this same PyTorch module into their own runtime
format -- the standard workflow for converter-based inference frameworks,
since none of them define models itself. MPS Backend is different in kind: it's
PyTorch's own execution backend (`torch.device("mps")`), not a converter,
so it reuses this exact module unconverted, just moved onto different
hardware -- see frameworks/mps_backend_adapter.py. See also
core/config.py's compatibility check and each frameworks/*_adapter.py
module's own docstring for which of these are execution-verified here
versus written-to-docs only.

Edge Impulse is listed here too, for a different reason than the
converters above: it has no local conversion path for this (or any)
PyTorch architecture at all (models come from Edge Impulse Studio, cloud,
account required), so `frameworks/edge_impulse_adapter.py`'s
`load_model()` always raises a clear, explanatory `RuntimeError` for
ResNet18 specifically -- but that error needs to actually be reached to
be useful. Leaving Edge Impulse out of `also_supports` would make
`core/config.py`'s `validate()` reject this combo one step earlier with
its own generic "architecture doesn't support this framework" message,
burying the more informative one -- listed here so `validate()` passes
and the real, intentional error surfaces where it's supposed to.
"""
from core.registry import ARCHITECTURES


def build(framework_adapter, config):
    import torchvision

    return torchvision.models.resnet18(weights=None, num_classes=10)


ARCHITECTURES.register(
    "ResNet18",
    implemented=True,
    family="CNN",
    framework="PyTorch",
    also_supports=[
        "TensorRT",
        "OpenVINO",
        "TensorFlow Lite",
        "ONNX Runtime",
        "ONNX Runtime Mobile",
        "PyTorch Mobile",
        "ExecuTorch",
        "TVM",
        "CoreML",
        "MPS Backend",
        "TensorFlow",
        "TensorFlow Lite Micro",
        "MediaPipe",
        "NCNN",
        "SNPE",
        "RKNN",
        "Arm NN",
        "DeepStream",
        "MNN",
        "NNAPI",
        "Edge Impulse",
    ],
    application="Image Classification",
    input_shape=(3, 32, 32),
)(build)
