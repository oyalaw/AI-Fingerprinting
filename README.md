# AI Fingerprinting Research Testbed

One codebase that generates real, labeled AI-workload network traffic for
encrypted-traffic fingerprinting research. It walks Paradigm -> Role ->
Device -> Framework -> Family -> Architecture -> Application -> Dataset ->
Transport (via config file or an interactive prompt), executes the real
workload, captures the resulting traffic with scapy, and auto-saves ground
truth labels for a 5-level classification taxonomy (Framework / Family /
Architecture / Application / Device).

Every option from the project's framework/architecture/dataset/device
tables is registered and visible via `--list`, even before it has a real
implementation behind it -- unimplemented entries are clearly marked
`[stub]` and raise a clear error if you try to run them. Currently
implemented end-to-end:

- **Inference**: PyTorch -> CNN -> ResNet18 -> Image Classification -> CIFAR10
- **Inference**: OpenVINO -> CNN -> ResNet18, same combo -- converts the PyTorch module straight to OpenVINO IR and compiles it for `CPU` by default. Runs on any x86/ARM machine, no special hardware needed (this one's verified end-to-end, including a real client/server roundtrip).
- **Inference (Jetson only)**: TensorRT -> CNN -> ResNet18, same combo -- exports the PyTorch module to ONNX and compiles a TensorRT engine. Needs an actual NVIDIA GPU (`tensorrt` + `pycuda`); validates and lists fine without them, it just can't build/run an engine on non-NVIDIA hardware.
- **Inference**: TensorFlow Lite -> CNN -> ResNet18, same combo -- converts the PyTorch module to a `.tflite` flatbuffer via `litert-torch` (Google's PyTorch->TFLite converter), then runs it through the standard TFLite interpreter API. Needs `litert-torch` (pulls in full `tensorflow`) plus `ai-edge-litert` for inference; both are normal pip installs but don't yet have wheels for every Python version -- validates and lists fine either way, conversion just needs a Python version `tensorflow` supports.
- **Inference**: ONNX Runtime -> CNN -> ResNet18, same combo -- exports the PyTorch module to ONNX (same export step TensorRT uses) and runs it through `onnxruntime.InferenceSession`. Pure Python/C++ wheel, no special hardware needed (this one's verified end-to-end too, including a real client/server roundtrip).
- **Inference**: ONNX Runtime Mobile -> CNN -> ResNet18, same combo -- runs the same ONNX export through ORT's real mobile model-prep pipeline (`onnxruntime.tools.convert_onnx_models_to_ort`, the API behind the official conversion CLI) to produce the same size/latency-optimized `.ort` flatbuffer format a real Android/iOS app would ship, then loads and runs it via `InferenceSession` on `CPUExecutionProvider` as a stand-in for the on-device NNAPI/CoreML execution provider a real mobile build would pick. Verified end-to-end here, including a real client/server roundtrip -- actual on-device NNAPI/CoreML execution needs a native Android/iOS app, out of scope for the same reason noted under Devices below.
- **Inference**: PyTorch Mobile -> CNN -> ResNet18, same combo -- `torch.jit.trace`s the module, runs Meta's real `optimize_for_mobile` pass, saves via `_save_for_lite_interpreter` (the actual `.ptl` mobile format), then loads it back with `_load_for_lite_interpreter` and runs inference on that loaded module -- no extra dependency at all, it's all in `torch`. Verified end-to-end, including a real client/server roundtrip; on a PyTorch build without XNNPACK (e.g. this project's own dev machine), `optimize_for_mobile` isn't available and the adapter falls back to saving the unoptimized traced module with a warning, rather than failing.
- **Inference (not execution-verified here)**: ExecuTorch -> CNN -> ResNet18, same combo -- Meta's newer PyTorch Mobile successor. `torch.export.export`s the module, lowers it with `executorch.exir.to_edge`, produces real `.pte` program bytes via `.to_executorch()`, then loads and runs that `.pte` through ExecuTorch's own documented host runtime (`_load_for_executorch`). Unlike every other adapter above, this one's blocked by dependency availability rather than hardware: `executorch` currently has no PyPI wheel for this project's Python version, so it validates and lists fine but hasn't actually been run -- treat frameworks/executorch_adapter.py as written-to-the-docs rather than verified, and re-check it against current ExecuTorch docs before relying on it.
- **Federated learning**: Flower (same PyTorch/ResNet18/CIFAR10 combo, partitioned across simulated clients)
- **Distributed training**: PyTorch `DistributedDataParallel` (same combo, multi-process gradient sync)
- **Transport**: TCP and TLS (self-signed dev cert auto-generated)

## Setup

```bash
pip install -r requirements.txt
```

`torch`/`torchvision`/`flwr` are large downloads. If you only want to browse
the registry (`--list`) or work on new adapters, `pyyaml` + `numpy` are
enough to get started.

### Packet capture prerequisites

Capture (`capture: true` in config, on by default) uses scapy and needs a
working packet-capture driver plus elevated privileges:

- **Windows**: install [Npcap](https://npcap.com/), then run your terminal
  as Administrator.
- **Linux / macOS**: libpcap is normally preinstalled; run with `sudo`, or
  grant your Python interpreter `CAP_NET_RAW` once so you don't need root
  every time:
  `sudo setcap cap_net_raw+eip $(readlink -f $(which python3))`

Without capture privileges, set `capture: false` in the config to still run
the workload and generate ground truth (just no `.pcap`/sequence CSV).

## Running the vertical slices

**Inference** (two terminals):

```bash
python main.py --config config.yaml --role server
python main.py --config config.yaml --role client
```

Switch `transport: TCP` to `transport: TLS` in `config.yaml` to compare
plaintext vs. encrypted traffic signatures.

**Federated learning** (one server + 2+ clients, each in its own terminal;
set `paradigm: federated_learning`, `fl_framework: Flower` in the config,
and give each client a distinct `client_index`):

```bash
python main.py --config config.yaml --role server
python main.py --config config.yaml --role client --client-index 0
python main.py --config config.yaml --role client --client-index 1
```

**Distributed training** (coordinator + 1+ workers; set
`paradigm: distributed_training`, `distributed_framework: DistributedDataParallel`):

```bash
python main.py --config config.yaml --role server               # rank 0 / coordinator
python main.py --config config.yaml --role client --worker-rank 1
```

**OpenVINO, TensorRT, TensorFlow Lite, ONNX Runtime, ONNX Runtime Mobile,
PyTorch Mobile, or ExecuTorch inference** (set `framework:` to `OpenVINO`,
`TensorRT`, `TensorFlow Lite`, `ONNX Runtime`, `ONNX Runtime Mobile`,
`PyTorch Mobile`, or `ExecuTorch` in the config, everything else stays the
same as the PyTorch inference example -- all but TensorRT run on any
machine, TensorRT needs an NVIDIA/Jetson GPU; ExecuTorch additionally
needs a Python version it ships a wheel for):

```bash
python main.py --config config.yaml --role server
python main.py --config config.yaml --role client
```

## Other commands

```bash
python main.py --list          # every registry entry, implemented vs. stub
python main.py --interactive   # guided step-by-step setup (Paradigm -> ... -> Transport)
```

## Output

Each run writes to `experiments/results/<experiment_id>/`:
`ground_truth.json` (the 5-level label + full config snapshot),
`<id>.pcap` (raw capture), `<id>_sequence.csv` (ordered
timestamp/direction/size sequence -- the standard feature representation
for traffic-fingerprinting classifiers), and `events.jsonl` / `experiment.log`.

## Extending

Every axis is a plugin registry (`core/registry.py`). To promote a stub to
a real implementation (e.g. a new mobile app for Android/iPhone, or one of
the other 21 stub frameworks), replace its `raise NotImplementedError` with
a real implementation and flip `implemented=False` to `True` in its
`.register(...)` call -- nothing else needs to change; auto-discovery picks
it up. `frameworks/tensorrt_adapter.py` is a worked example of promoting a
framework whose architectures aren't natively built in it (it consumes the
same PyTorch-built architectures via ONNX export -- see each architecture's
`also_supports` registration metadata and the compatibility check in
`core/config.py`).
