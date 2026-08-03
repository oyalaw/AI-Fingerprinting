"""ResNet18 -- the one fully-implemented architecture this pass. Random init
(not pretrained) so no network download is required at model-load time; the
point of this project is the traffic it generates, not classification
accuracy. num_classes=10 to match CIFAR10. torchvision is imported lazily
inside build() so registering this architecture doesn't require it
installed.

Natively built with PyTorch, but `also_supports` TensorRT, OpenVINO,
TensorFlow Lite, ONNX Runtime, ONNX Runtime Mobile, PyTorch Mobile,
ExecuTorch, TVM, CoreML, and MPS Backend: the first nine adapters
convert/compile this same PyTorch module into their own runtime format --
the standard workflow for converter-based inference frameworks, since
none of them define models itself. MPS Backend is different in kind: it's
PyTorch's own execution backend (`torch.device("mps")`), not a converter,
so it reuses this exact module unconverted, just moved onto different
hardware -- see frameworks/mps_backend_adapter.py. See also
core/config.py's compatibility check, frameworks/tensorrt_adapter.py,
frameworks/openvino_adapter.py, frameworks/tensorflow_lite_adapter.py,
frameworks/onnx_runtime_adapter.py, frameworks/onnx_runtime_mobile_adapter.py,
frameworks/pytorch_mobile_adapter.py, frameworks/executorch_adapter.py,
frameworks/tvm_adapter.py, and frameworks/coreml_adapter.py.
"""
from core.registry import ARCHITECTURES


def build(framework_adapter):
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
    ],
    input_shape=(3, 32, 32),
)(build)
